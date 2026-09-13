"""Reef coordination across independent training and inference operations.

Reef owns admission, job identity/replay, checkpoint ordering, publication
identity, adapter residency, resource handoff and fenced recovery. Backend
adapters implement data preparation, optimizer execution, checkpoint I/O and
native sender/receiver operations without controlling each other's lifecycle.
"""
