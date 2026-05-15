---
name: usda-export-script-generator
description: Use this skill when a user asks to add or modify a USDA export script for a concept template category. The agent must inspect existing exporters and concept templates, optionally search official USD/Omniverse physics docs, then implement a runnable `concept_templates/<Category>/export_*.py` script with configurable physics, joints, grasps, and validation guidance.
---

# USDA Export Script Generator Skill

## Mission

Use this skill to create or update USDA export scripts in this repository for any object category.

Typical targets:

- `concept_templates/<Category>/export_*_with_simple_collision.py`

The script should export a usable USD scene prim (defaultPrim set), with visual meshes, collision, physics attributes, and task-relevant metadata (for example grasps, joints, constraints).

## Core Behavior

### 1) Prompt-driven requirement parsing

Before coding, extract and restate:

- Object category and source assets (conceptualization pkl/json, segmented objs, textures)
- Export mode (synthetic concept mesh vs real segmented mesh)
- Physics behavior required (single rigid, multi-link, articulated, breakable joint, etc.)
- Output expectations (filename, root prim path, parameters, defaults)

If a requirement affects scene behavior and is ambiguous, ask targeted questions immediately.

### 2) Repository-first analysis

Always inspect existing patterns before implementation.

Minimum files to inspect:

- target category exporter files under `concept_templates/<Category>/`
- `concept_templates/<Category>/concept_template.py`
- `concept_templates/<Category>/knowledge_definitions.py`
- `concept_templates/<Category>/knowledge_utils.py`

Also inspect at least one mature reference exporter from another category when adding advanced behavior:

- multi-link rigid setup
- revolute/prismatic joints
- collision approximation tuning
- grasp metadata export

### 3) USD/Physics schema correctness

Use proper USD APIs instead of ad-hoc attributes whenever available:

- `UsdGeom` for prim hierarchy and transforms
- `UsdPhysics` for rigid bodies, mass, collision, joints, drives
- `UsdShade` for visual/physics materials

When behavior is non-trivial (joint limits, break force/torque, articulation exclusion, etc.), verify against official OpenUSD/Omniverse docs before finalizing.

### 4) Evidence-based API selection

Before substantial code edits, gather short evidence:

- searched keywords
- repository references reused
- external API references used
- why selected implementation matches requested behavior

## Non-Negotiable Workflow

Follow this sequence:

1. Parse prompt into export behavior spec
2. Read target category templates and existing exporters
3. Check official USD/Omniverse docs for unstable/advanced APIs
4. Send pre-coding alignment message
5. Wait for user confirmation if major behavior tradeoffs exist
6. Implement script and parameters
7. Run local checks (syntax/static checks)
8. Provide run instructions and tuning notes

## Mandatory Pre-Coding Alignment Message

Before editing code, send:

1. Understanding
- requested export behavior
- expected prim hierarchy and physics model
- required assets and metadata

2. Plan
- file(s) to create/update
- helper functions to add/reuse
- parameter surface and defaults
- verification steps

3. Open questions
- unresolved assumptions only

## Implementation Standards

### A) Script structure

Organize exporters into:

- asset loading/parsing helpers
- mesh authoring helpers (visual/collision)
- physics helpers (rigid body, material, joints)
- metadata helpers (grasp poses, semantic attributes)
- main export function with explicit parameters
- `main()` entry for local execution

### B) Prim hierarchy conventions

Use clear, stable hierarchy:

- root object prim (defaultPrim)
- optional `links` and `joints` subtrees for articulated objects
- separate `visual` and `collision` meshes per link
- optional `Looks` scope for materials

### C) Physics and collision conventions

- Use separate rigid bodies for independently movable parts
- Apply collision APIs on collision meshes only
- Set collision approximation explicitly (`convexDecomposition`, etc.)
- Keep masses and friction configurable via function args
- For articulated behavior, set joint limits/drives/break parameters as args

### D) Grasp/interaction metadata

When category knowledge supports it, export candidate grasp transforms as prims or attributes.

At minimum include:

- grasp pose transform
- approach direction
- optional transformation matrix array for downstream planners

### E) Parameterization

Expose all behavior-critical values as args with defaults:

- `scale_to_meters`
- masses
- friction/restitution
- initial pose
- joint limits, damping/stiffness, break thresholds

Avoid hard-coding values in body logic.

## Validation Requirements

Always perform and report:

1. Syntax check
- for python exporters, run compile check (`python3 -m py_compile ...`)

2. Structural sanity
- confirm root/default prim
- confirm expected prim paths exist in generated logic

3. Runtime guidance
- how to run script
- output USDA location
- what to verify in Isaac/Omniverse (pose, collision, articulation behavior)

If GUI/runtime validation cannot run in this environment, state that clearly.

## User Collaboration Rules

Ask concise clarifications only for high-impact ambiguity:

- exact output filename/path
- requested articulated behavior details
- success expectation (for example rotation range before detaching)
- whether backward compatibility with old exporter must be preserved

Otherwise proceed with reasonable defaults and state them.

## Deliverable Checklist

Before final response, verify:

1. New/updated exporter file created
2. Functional export entrypoint implemented
3. Requested physics behavior encoded
4. Parameters documented in code and summary
5. Local syntax check run and result reported
6. Run instructions and tuning knobs provided

## Suggested Search Keywords

- `OpenUSD UsdPhysics RevoluteJoint`
- `OpenUSD UsdPhysics Joint breakForce breakTorque`
- `Omniverse USD physics collision approximation`
- `Isaac Sim USD articulation joint limits python`
- `<Category> export usda python`

## Primary References

- https://openusd.org/release/api/
- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/
- https://docs.isaacsim.omniverse.nvidia.com/

## Local Configuration

- ubuntu 22.04
- zsh
- isaac 4.0.0

## Tips

- Prefer reusing helper patterns from existing category exporters over rewriting from scratch.
- Keep transforms explicit and unit-consistent (`metersPerUnit = 1.0`, controlled scaling).
- For detachable parts, model initial attachment with a joint and detach via break thresholds instead of single rigid-body hacks.
