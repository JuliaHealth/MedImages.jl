---
name: V-Model SDLC Orchestrator
description: Orchestrate the V-Model Software Development Lifecycle via Gemini CLI and Agentic Skill Frameworks. Enforce rigor, traceability, and structural integrity in engineering.
---

# V-Model SDLC Orchestrator Skill

This skill empowers the AI agent to act as a rigorous software architect and engineer, following the V-Model methodology (Validation & Verification).

## Core Philosophy
The V-Model requires a one-for-one relationship between development phases and their corresponding testing levels. This skill prevents "vibe coding" by enforcing sequential integrity and traceability.

## V-Model Mapping

| SDLC Phase | Gemini / Spec-Kit Command | Primary Artifact | Verification Counterpart |
| :--- | :--- | :--- | :--- |
| **Requirements Analysis** | `/speckit.specify` | `PRD.md` / `SRS.md` | **Acceptance Testing** |
| **System Design** | `analyze_codebase`, `mermaid` | `Arch_Diagrams.md` | **System Testing** |
| **Architecture Design** | `estimate_effort` | `Tech_Spec.md` | **Integration Testing** |
| **Module Design** | `/speckit.plan` | `Module_Specs.md` | **Unit Testing** |
| **Coding** | `/speckit.implement` | Source Code | **Static Analysis** |

## Procedures

### 1. Phase Identification & Gatekeeping
Before starting any task, identify which phase of the V-Model you are in.
- **Rule**: Never write implementation code before a matching unit test is approved in the Module Design phase.
- **Rule**: Always link code changes to a specific Requirement ID from the PRD.

### 2. Requirements & Specification
Use `/speckit.specify` to capture the "what" and "why".
- Automate the transition from raw notes to `PRD.md`.
- Ensure requirements are testable and unambiguous.

### 3. Traceability Matrix Maintenance
Maintain a Compliance Traceability Matrix (CTM) in `docs/traceability_matrix.md`.
- Map project principles to implementation artifacts.
- Use `speckit.analyze` to cross-artifact consistency.

### 4. Verification & Self-Healing
If a test or CI/CD build fails:
- Act as a self-healing agent.
- Read logs via MCP, analyze the failure, and open a PR with the fix.
- Address flaky tests by analyzing brittle/victim logic.

### 5. Multi-Agent Review
When performing complex architectural tasks, simulate or request peer review from another reasoning-heavy model (e.g., Claude or Codex) via the `.ai` folder symlink framework to avoid silos.

## Governance
- **Self-Mode**: Favor "Safe Mode" where every file mod requires confirmation.
- **DNA Files**: Respect `GEMINI.md` and `.specify/constitution.md` for project-specific constraints.
