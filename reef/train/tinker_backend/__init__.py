"""Optional Tinker integration: remote LoRA training and immutable sampling.

Deployment/config discovery does not import the SDK. ``client`` owns the SDK
boundary, ``losses`` names the Tinker loss and shapes its inputs, and
``runtime`` owns snapshots, publication, and the chat inference backend.
"""
