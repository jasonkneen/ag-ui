# Changelog

## 1.0.0 — 2026-09-17

- Client now declares the protocol version it speaks: every `RunAgentInput` carries "1.0", exposed as `AGUIChatClient.WireProtocolVersion`.
- Client validates the producer's protocol version, mirroring the TypeScript client handshake.
- Moved onto the generated 1.0 models.

### Breaking changes

- The hand-written `WIRE_PROTOCOL_VERSION` constant is removed; use the generated `PROTOCOL_VERSION` and `AGUIChatClient.WireProtocolVersion`.
- Wire protocol version now sent as "1.0"; model names and null handling follow the 1.0 schema.
