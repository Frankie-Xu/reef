// Harness requests: the /reef-harness and /reef-versions commands and the
// reef_ask_user and reef_file_request tools for reef-pi. The person asks in
// plain words. With a UI the session model first thinks the request through,
// asks what is unclear (reef_ask_user) and files it (reef_file_request); with
// --direct, or headless, the command files it as is. A filed request goes to
// reef with this session's id and the installed release through native manual
// training; the service proposer writes the change, and a watch here polls the
// catalog, shows in the footer whether the request is queued or its step is
// running and for how long, and reports the step's verdict in the session,
// with why the proposer produced nothing when it did. /reef-versions lists
// the release chain with each step's verdict and request, prints a step's page
// and, for a pending release, the promote action and a trial install, and runs
// the promote after a confirmation. Nothing here writes a mutation. Kept free
// of annotations on purpose: plain JavaScript in a .ts file, so plain node can
// parse it in CI and pi's TS loader accepts it unchanged. Gate episodes set
// PI_OFFLINE and this extension then registers nothing, so the gate never sees
// the commands or the tools.
import { readFileSync } from "node:fs";
import { join } from "node:path";

// The release file the install script and harness_pull write at the tree root.
const RELEASE_FILE = ".reef-harness-release";
// The watch polls the catalog (and, until a step takes the request, its record) once per interval and gives up
// at the cap.
const WATCH_INTERVAL_MS = 5000;
const WATCH_CAP_MS = 30 * 60 * 1000;
// The service caps a request's text; the filed text stays within it.
const REQUEST_MAX_CHARS = 4000;
// The choice under every question that opens a free text answer.
const OTHER = "Other (type an answer)";
const NO_UI_TEXT = "no UI in this session: proceed with your best assumptions and list them in the request";

// Tool parameters as plain JSON schema: pi compiles them with typebox, which reads JSON schema as is, so the
// extension needs no import beyond node.
const ASK_USER_PARAMETERS = {
  type: "object",
  properties: {
    questions: {
      type: "array",
      minItems: 1,
      maxItems: 4,
      items: {
        type: "object",
        properties: {
          question: { type: "string", description: "one open point, as a question" },
          options: { type: "array", items: { type: "string" }, minItems: 2, maxItems: 4 },
        },
        required: ["question", "options"],
      },
    },
  },
  required: ["questions"],
};
const FILE_REQUEST_PARAMETERS = {
  type: "object",
  properties: {
    request: { type: "string", description: "the user's original words" },
    clarifications: {
      type: "array",
      items: {
        type: "object",
        properties: { question: { type: "string" }, answer: { type: "string" } },
        required: ["question", "answer"],
      },
    },
  },
  required: ["request"],
};

function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

function message(error) {
  return error instanceof Error ? error.message : String(error);
}

// The first `limit` characters of a text, the cut marked.
function clip(text, limit) {
  return text.length > limit ? `${text.slice(0, limit - 3)}...` : text;
}

// The poll interval, from the environment so a test can shorten it; the cap stays.
function watchIntervalMs() {
  const configured = Number(process.env.REEF_HARNESS_WATCH_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : WATCH_INTERVAL_MS;
}

// What the session model does with a request before it files it: the request rides as data in a fence.
function clarifyMessage(text) {
  return [
    "The user asked for this harness change:",
    "",
    "```",
    text,
    "```",
    "",
    "Before filing it with reef_file_request, think through what it needs: when it triggers, what state the " +
      "harness must know and how it learns it, what the user must set up, what is ambiguous. If an open point " +
      "would change what gets built, ask with reef_ask_user: at most 3 questions, each with 2 to 4 concrete " +
      "options; the user can always type their own. Then call reef_file_request with the user's original words " +
      "as `request` and the answers as `clarifications`. Do not write the change yourself: reef's service writes it.",
  ].join("\n");
}

// The filed text: the request verbatim, then the answers as question and answer pairs.
function filedText(request, clarifications) {
  const pairs = (Array.isArray(clarifications) ? clarifications : []).filter(
    (item) => item && typeof item.question === "string" && typeof item.answer === "string",
  );
  const lines = [request.trim()];
  if (pairs.length) lines.push("", "Clarifications:", ...pairs.map((item) => `- Q: ${item.question}\n  A: ${item.answer}`));
  return lines.join("\n").slice(0, REQUEST_MAX_CHARS);
}

// The metrics a catalog row carries, or nothing.
function metricsOf(row) {
  return row && row.metrics && typeof row.metrics === "object" ? row.metrics : {};
}

// The record id of the request a step consumed; the watch keys its row by it.
function requestIdOf(row) {
  const request = metricsOf(row).training_request;
  return request && typeof request.id === "string" ? request.id : null;
}

// What the review left uncovered, when the step recorded a review.
function uncoveredOf(row) {
  const notes = metricsOf(row).proposal_notes;
  const review = notes && typeof notes === "object" ? notes.review : null;
  const items = review && Array.isArray(review.uncovered) ? review.uncovered : [];
  return items.filter((item) => typeof item === "string" && item.trim()).map((item) => item.trim());
}

// Why the proposer produced nothing, when the step recorded it beside its notes.
function failureOf(row) {
  const notes = metricsOf(row).proposal_notes;
  return notes && typeof notes === "object" && typeof notes.failure === "string" ? notes.failure.trim() : "";
}

// An elapsed time as the footer shows it: minutes and two digit seconds.
function elapsedText(ms) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

export default function requests(pi) {
  if (process.env.PI_OFFLINE) return; // hermetic episodes never see the commands or the tools
  const agentDir = process.env.PI_CODING_AGENT_DIR;
  const serviceUrl = process.env.REEF_SERVICE_URL;
  const scenario = process.env.REEF_SCENARIO;
  if (!agentDir || !serviceUrl || !scenario) return;
  // The wrapper relocates the agent into a temp copy and exports the true
  // install root; a tree run directly falls back to the release file beside it.
  const destDir = process.env.REEF_HARNESS_DEST || join(agentDir, "..");

  const reefHeaders = () => {
    const token = process.env.REEF_TOKEN;
    return { "x-reef-scenario": scenario, ...(token ? { authorization: `Bearer ${token}` } : {}) };
  };

  const installedRelease = () => {
    const releaseInfo = readJson(join(destDir, RELEASE_FILE));
    return releaseInfo && typeof releaseInfo.release_id === "string" && releaseInfo.release_id ? releaseInfo.release_id : null;
  };

  const noReleaseText = () =>
    `no ${RELEASE_FILE} release file at ${destDir}: this tree did not come through reef's install channel, ` +
    "so a request cannot name the release it runs; nothing was sent";

  // POST the request with this session and the installed release; the answer names the record the step's
  // catalog row carries. Throws with the message the notice shows.
  const fileRequest = async (text, ctx) => {
    const releaseId = installedRelease();
    if (!releaseId) throw new Error(noReleaseText());
    const body = { text, session: ctx.sessionManager.getSessionId(), release_id: releaseId };
    let response;
    try {
      // Not under the turn's abort signal: an Esc after the body went out would report a filed request as unreachable.
      response = await fetch(`${serviceUrl}/reef/train`, {
        method: "POST",
        headers: { ...reefHeaders(), "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (error) {
      throw new Error(`reef unreachable at ${serviceUrl}: ${message(error)}`);
    }
    if (!response.ok) throw new Error(`reef refused the request (HTTP ${response.status}): ${await response.text()}`);
    const answer = await response.json();
    return String(answer.agent_record_id);
  };

  // The catalog oldest first; a step is a row's position in it, the creation row being 0, which is the commit
  // step the service keys the page by (a rejected step publishes nothing, so only its position names it).
  const releases = async () => {
    let response;
    try {
      response = await fetch(`${serviceUrl}/reef/harness/releases`, { headers: reefHeaders() });
    } catch (error) {
      throw new Error(`reef unreachable at ${serviceUrl}: ${message(error)}`);
    }
    if (!response.ok) throw new Error(`reef refused the catalog read (HTTP ${response.status}): ${await response.text()}`);
    const rows = (await response.json()).releases;
    return Array.isArray(rows) ? rows : [];
  };

  // The request's own record: its compacted_at is null while the request is queued and a time once a step
  // took it, which is when the wait a person sees starts.
  const requestRecord = async (recordId) => {
    const path = `/reef/scenarios/${encodeURIComponent(scenario)}/records/${encodeURIComponent(recordId)}`;
    const response = await fetch(`${serviceUrl}${path}`, { headers: reefHeaders() });
    if (!response.ok) throw new Error(`reef refused the record read (HTTP ${response.status})`);
    return await response.json();
  };

  // A promoted row stays pending in the catalog; the promote is a later row naming it, so with the rows given
  // the pending row reads "promoted at step N".
  const verdictOf = (row, rows = []) => {
    if (row.pending) {
      const promoted = rows.findIndex(
        (other) => other.operation === "promote" && other.rollback_target_release_id === row.release_id,
      );
      return promoted >= 0 ? `promoted at step ${promoted}` : "pending";
    }
    const metrics = metricsOf(row);
    if (typeof metrics.selected === "boolean") return metrics.selected ? "selected" : "rejected";
    if (metrics.skipped) return "skipped";
    return String(row.operation || "unknown");
  };

  // The one line a settled step earns, with the next action, quoting the request; the wrapper prints the same.
  // Every line names the step, whose page holds the details.
  const settledText = (step, rows, ask) => {
    const row = rows[step];
    const metrics = metricsOf(row);
    const release = String(row.release_id || "").slice(0, 8);
    const verdict = verdictOf(row, rows);
    const details = ` Details: /reef-versions ${step}.`;
    if (verdict === "selected") {
      return (
        `reef: '${ask}' is published as release ${release}. Restart reef-pi to install it (the update notice ` +
        `offers it).${details}`
      );
    }
    if (verdict === "pending") {
      return (
        `reef: '${ask}' is ready as release ${release} but changes an extension, so it waits for your review: ` +
        `/reef-versions ${step}, then /reef-versions ${step} promote.`
      );
    }
    if (verdict === "rejected") {
      const reason = metrics.selection && metrics.selection.reason ? metrics.selection.reason : "no reason recorded";
      return (
        `reef: '${ask}' did not pass the gate (${reason}). Nothing changed; rephrase or split the request.` + details
      );
    }
    if (verdict === "skipped") {
      // The proposer's own reason, when the step recorded one: a failed model call, a reply with no entry.
      const failure = failureOf(row);
      const why = failure ? `${metrics.skipped}: ${failure}` : String(metrics.skipped);
      return `reef: '${ask}' produced no change (${why}). Nothing changed.${details}`;
    }
    return `reef: '${ask}' settled as ${verdict} (release ${release}); /reef-versions ${step} shows it.`;
  };

  // The watch: one at a time, so a second filing replaces the first; session_shutdown clears it.
  let watch = null;

  const stopWatch = (ctx) => {
    if (!watch) return;
    clearInterval(watch.timer);
    watch = null;
    ctx.ui.setStatus("reef", undefined);
  };

  const startWatch = (recordId, text, ctx) => {
    stopWatch(ctx);
    const ask = clip(text.trim(), 60);
    const id8 = recordId.slice(0, 8);
    const deadline = Date.now() + WATCH_CAP_MS;
    // startedAt is the first poll that saw a step holding the request; the footer counts from it.
    const mine = { timer: null, polling: false, startedAt: null, status: null };
    const show = (status) => {
      if (status === mine.status) return; // the footer is redrawn only when its text changes
      mine.status = status;
      ctx.ui.setStatus("reef", status);
    };
    const tick = async () => {
      let rows = [];
      try {
        rows = await releases();
      } catch {
        // A failed read is one missed poll; the next tick reads again.
      }
      if (watch !== mine) return; // replaced or shut down while the read was out
      const step = rows.findIndex((row) => requestIdOf(row) === recordId);
      if (step >= 0) {
        stopWatch(ctx);
        const uncovered = uncoveredOf(rows[step]);
        const lines = [settledText(step, rows, ask)];
        if (uncovered.length) lines.push(`Not covered: ${uncovered.join("; ")}`);
        ctx.ui.notify(lines.join("\n"), "info");
        return;
      }
      if (Date.now() >= deadline) {
        stopWatch(ctx);
        ctx.ui.notify(`reef: no verdict yet for '${ask}'; /reef-versions shows it when it settles`, "warning");
        return;
      }
      if (mine.startedAt === null) {
        try {
          const record = await requestRecord(recordId);
          if (record && typeof record.compacted_at === "number") mine.startedAt = Date.now();
        } catch {
          // A failed record read keeps the footer as it was; the next tick reads again.
        }
        if (watch !== mine) return;
      }
      if (mine.startedAt !== null) {
        show(`reef: step for request ${id8} running for ${elapsedText(Date.now() - mine.startedAt)}`);
      }
    };
    const poll = async () => {
      if (mine.polling) return; // a slow read never overlaps the next tick
      mine.polling = true;
      try {
        await tick();
      } finally {
        mine.polling = false;
      }
    };
    mine.timer = setInterval(poll, watchIntervalMs());
    // A headless session exits when its turn ends; the timer must not hold the process open for the verdict.
    if (typeof mine.timer.unref === "function") mine.timer.unref();
    watch = mine;
    show(`reef: request ${id8} queued`);
  };

  pi.on("session_shutdown", async (_event, ctx) => stopWatch(ctx));

  pi.registerTool({
    name: "reef_ask_user",
    label: "Ask the user",
    description:
      "Ask the user up to 4 questions before filing a harness change with reef_file_request. Each question " +
      "offers 2 to 4 concrete options; the user can always type an answer of their own.",
    parameters: ASK_USER_PARAMETERS,
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      if (!ctx.hasUI) return { content: [{ type: "text", text: NO_UI_TEXT }], details: {} };
      const answers = [];
      for (const item of params.questions) {
        const choice = await ctx.ui.select(item.question, [...item.options, OTHER]);
        const answer = choice === undefined || choice === OTHER ? await ctx.ui.input(item.question, "") : choice;
        answers.push({ question: item.question, answer: answer === undefined ? "no answer" : answer });
      }
      return { content: [{ type: "text", text: JSON.stringify(answers) }], details: {} };
    },
  });

  pi.registerTool({
    name: "reef_file_request",
    label: "File a harness request",
    description:
      "File a harness change with reef: the user's original request and the answers reef_ask_user collected. " +
      "Reef's service writes the change and reports here when the step settles.",
    parameters: FILE_REQUEST_PARAMETERS,
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const text = filedText(params.request, params.clarifications);
      const recordId = await fileRequest(text, ctx); // a failure throws: the model reads the message
      startWatch(recordId, text, ctx);
      return {
        content: [
          {
            type: "text",
            text:
              `filed request ${recordId}; reef is running the step, which usually takes one to three minutes, ` +
              "and will report here when it settles",
          },
        ],
        details: {},
      };
    },
  });

  pi.registerCommand("reef-harness", {
    description: "Ask reef to grow this harness: /reef-harness [--direct] <what it should do>",
    handler: async (args, ctx) => {
      const words = (args || "").trim();
      const direct = words === "--direct" || words.startsWith("--direct ");
      const text = (direct ? words.slice("--direct".length) : words).trim();
      if (!text) {
        ctx.ui.notify("Usage: /reef-harness <what the harness should do>", "warning");
        return;
      }
      if (!installedRelease()) {
        ctx.ui.notify(noReleaseText(), "error");
        return;
      }
      if (!direct && ctx.hasUI) {
        // The session model asks what is unclear, then files through the tool; a busy agent takes it as a follow up.
        pi.sendUserMessage(clarifyMessage(text), ctx.isIdle() ? undefined : { deliverAs: "followUp" });
        ctx.ui.notify("reef: clarifying, then filing", "info");
        return;
      }
      let recordId;
      try {
        recordId = await fileRequest(text, ctx);
      } catch (error) {
        ctx.ui.notify(message(error), "error");
        return;
      }
      ctx.ui.notify(`Training request ${recordId} accepted; the step usually takes one to three minutes.`, "info");
      startWatch(recordId, text, ctx);
    },
  });

  // The served head: the newest row that is neither pending nor a rejected or skipped step, since those publish
  // nothing and carry the head's id. The catalog's own current flag sits on the newest row, whatever it is.
  const headStep = (rows) => {
    for (let index = rows.length - 1; index >= 0; index--) {
      if (!["pending", "rejected", "skipped"].includes(verdictOf(rows[index]))) return index;
    }
    return -1;
  };

  const requestText = (row) => {
    const request = metricsOf(row).training_request;
    const text = request && typeof request.text === "string" ? request.text.trim() : "";
    return text ? `"${clip(text, 60)}"` : "";
  };

  const lineOf = (step, rows) =>
    [
      String(step),
      String(rows[step].release_id || "").slice(0, 8),
      verdictOf(rows[step], rows),
      step === headStep(rows) ? "current" : "",
      requestText(rows[step]),
    ]
      .filter(Boolean)
      .join("  ");

  // What the proposer planned and what its review left uncovered, when the step recorded them.
  const notesLines = (row) => {
    const notes = metricsOf(row).proposal_notes;
    const design = notes && typeof notes.design === "string" ? notes.design.trim() : "";
    const lines = design ? [`design: ${clip(design, 200)}`] : [];
    const uncovered = uncoveredOf(row);
    if (uncovered.length) lines.push(`not covered: ${uncovered.join("; ")}`);
    return lines;
  };

  // What the step's model calls cost in tokens, when the endpoint reported them: the proposer's and the gate's.
  const tokenText = (row) => {
    const metrics = metricsOf(row);
    const over = (side, key) =>
      Object.values(side || {}).reduce((total, agent) => total + (Number(agent && agent[key]) || 0), 0);
    const proposerIn = Number(metrics.proposer_input_tokens) || 0;
    const proposerOut = Number(metrics.proposer_output_tokens) || 0;
    const gateIn = over(metrics.candidate_agents, "input_tokens") + over(metrics.current_agents, "input_tokens");
    const gateOut = over(metrics.candidate_agents, "output_tokens") + over(metrics.current_agents, "output_tokens");
    if (!proposerIn && !proposerOut && !gateIn && !gateOut) return "";
    return `tokens: proposer ${proposerIn} in / ${proposerOut} out, gate ${gateIn} in / ${gateOut} out`;
  };

  const pageUrl = (step) => `${serviceUrl}/reef/harness/releases/${step}/page`;
  // The token stays in the environment: the printed command names it as the variable, never its value.
  const curl = () =>
    `curl -fsS -H 'x-reef-scenario: ${scenario}' ` + (process.env.REEF_TOKEN ? '-H "Authorization: Bearer $REEF_TOKEN" ' : "");

  const installLine = (releaseId) =>
    `${curl()}'${serviceUrl}/reef/harness/install?adapter=pi&release_id=${encodeURIComponent(releaseId)}'` +
    ` | bash -s -- '${destDir}'`;

  const stepLines = (step, rows) => {
    const row = rows[step];
    const head = headStep(rows);
    const verdict = verdictOf(row, rows);
    const lines = [
      `Harness step ${step}: ${row.release_id} (${verdict}${step === head ? ", current" : ""})`,
      ...notesLines(row),
      `page: ${pageUrl(step)}`,
      `read it: ${curl()}'${pageUrl(step)}' > harness-step-${step}.html`,
    ];
    if (verdict === "pending") {
      const body = JSON.stringify({ release_id: row.release_id });
      lines.push(
        `promote: ${curl()}-X POST -H 'content-type: application/json' -d '${body}' ` +
          `'${serviceUrl}/reef/scenarios/${encodeURIComponent(scenario)}/promote'`,
        `or from here: /reef-versions ${step} promote`,
        `trial install (replaces the tree at ${destDir}): ${installLine(row.release_id)}`,
      );
      if (head >= 0) lines.push(`back to the head: ${installLine(rows[head].release_id)}`);
    }
    const tokens = tokenText(row);
    if (tokens) lines.push(tokens);
    return lines;
  };

  pi.registerCommand("reef-versions", {
    description: "List this harness's versions, or show one: /reef-versions [step] [promote]",
    handler: async (args, ctx) => {
      const words = (args || "").trim().split(/\s+/).filter(Boolean);
      const promote = words[1] === "promote";
      // Digits only before Number(): "1e1" and "0x3" are numbers to it and no step to the catalog.
      const usable = words.length === 0 || (/^\d+$/.test(words[0]) && (words.length === 1 || (promote && words.length === 2)));
      if (!usable) {
        ctx.ui.notify("Usage: /reef-versions [step] [promote]", "warning");
        return;
      }
      const step = words.length ? Number(words[0]) : null;
      let rows;
      try {
        rows = await releases();
      } catch (error) {
        ctx.ui.notify(message(error), "error");
        return;
      }
      if (step === null) {
        ctx.ui.notify(rows.length ? rows.map((_, index) => lineOf(index, rows)).join("\n") : "no release on record", "info");
        return;
      }
      const row = rows[step];
      if (!row) {
        ctx.ui.notify(`no step ${step}: the catalog holds steps 0 to ${rows.length - 1}`, "warning");
        return;
      }
      if (!promote) {
        ctx.ui.notify(stepLines(step, rows).join("\n"), "info");
        return;
      }
      const verdict = verdictOf(row, rows);
      if (verdict.startsWith("promoted")) {
        ctx.ui.notify(`step ${step} is already ${verdict}; nothing to promote`, "warning");
        return;
      }
      if (verdict !== "pending") {
        ctx.ui.notify(`step ${step} is not pending (${verdict}); nothing to promote`, "warning");
        return;
      }
      const confirmed = await ctx.ui.confirm(
        `Promote harness step ${step}?`,
        `Release ${row.release_id} then serves every session that installs the head. Read ${pageUrl(step)} first.`,
      );
      if (!confirmed) {
        ctx.ui.notify(`step ${step} not promoted`, "info");
        return;
      }
      let response;
      try {
        response = await fetch(`${serviceUrl}/reef/scenarios/${encodeURIComponent(scenario)}/promote`, {
          method: "POST",
          headers: { ...reefHeaders(), "content-type": "application/json" },
          body: JSON.stringify({ release_id: row.release_id }),
        });
      } catch (error) {
        ctx.ui.notify(`reef unreachable at ${serviceUrl}: ${message(error)}`, "error");
        return;
      }
      if (!response.ok) {
        ctx.ui.notify(`reef refused the promote (HTTP ${response.status}): ${await response.text()}`, "error");
        return;
      }
      const answer = await response.json();
      ctx.ui.notify(
        `Promoted step ${step}: the head is now ${answer.release_id}; the update notice offers it at the next session start.`,
        "info",
      );
    },
  });

  // Said once per session start with a UI: the two commands exist, and what waits for a review.
  pi.on("session_start", async (_event, ctx) => {
    if (!ctx.hasUI) return;
    const lines = ["reef: /reef-harness <what it should do> asks for a harness change; /reef-versions lists the versions."];
    try {
      const rows = await releases();
      const waiting = rows.map((row, step) => (verdictOf(row, rows) === "pending" ? step : -1)).filter((step) => step >= 0);
      if (waiting.length) lines.push(`${waiting.length} release(s) await your review: /reef-versions ${waiting.join(", ")}`);
    } catch {
      // The catalog is a courtesy here: the first line stands without it.
    }
    ctx.ui.notify(lines.join("\n"), "info");
  });
}
