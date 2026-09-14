"""Optional Tinker integration: remote LoRA training and immutable sampling.

Deployment/config discovery does not import the SDK. ``client`` owns the SDK
boundary, ``losses`` names the Tinker loss and shapes its inputs, ``runtime``
owns the checkpoint store and the training runtime, and ``inference`` owns the
inference runtime and its chat handler. The launch factory returns the two
runtimes as a pair over one store.
"""
