---
name: isaac-universal-task-generator
description: Use this skill when a user asks to implement Isaac Sim + cuRobo manipulation tasks for any object category and any operation type. The agent must infer object/task requirements from the user prompt, analyze repository concept templates and existing task implementations, then generate paired files `actions/franka_*_action.py` and `actions/test_in_isaac_*.py` with evaluation metrics and Isaac validation guidance.
---

# Isaac Sim Universal Task Generator Skill

## Mission

Use this skill to implement manipulation tasks in this repository for **any object** and **any task type** (for example grasp, place, pull, push, open, close, rotate, insert, handover, etc.).

The agent must parse the user prompt to determine:

- target object category and instance
- target operation/task
- task success criteria
- required output file names

Default output pair:

- `actions/franka_*_action.py`
- `actions/test_in_isaac_*.py`

## Core Behavior

### 1) Prompt-driven task/object inference

Before coding, extract and restate:

- Object: which object category and which part is manipulated
- Task: what physical state change is expected
- Constraints: deterministic execution, speed, collision assumptions, scene assumptions
- Deliverables: exact action/test file names

If any task-critical parameter is ambiguous, ask targeted questions immediately.

### 2) Repository-first understanding

Always analyze repository abstractions before implementation.

Minimum files to inspect:

- `actions/franka_grasp_and_place_action.py`
- `actions/test_in_isaac_grasp_place.py`
- `actions/scene_object_loader.py`
- relevant `concept_templates/<ObjectCategory>/concept_template.py`
- relevant `concept_templates/<ObjectCategory>/knowledge_definitions.py`
- relevant `concept_templates/<ObjectCategory>/knowledge_utils.py`

For multi-object tasks, inspect all involved categories.

### 3) Concept-template interpretation requirement

Understand and explicitly use this repository’s object abstraction:

- One object category is decomposed into multiple concept templates (parts/components).
- Each template encodes geometry + semantics + functional knowledge.
- Knowledge files provide affordance/manipulation logic (not only mesh generation).

When presenting your understanding, cite concrete classes/functions you will reuse.

### 4) External evidence search before coding

Search for reusable references in this order:

1. Isaac Sim / cuRobo open-source examples
2. Official Isaac Sim docs
3. Official cuRobo docs/repo

Decision rule:

- Reuse/adapt robust existing examples if they match the task.
- Otherwise implement from scratch using official APIs and repository conventions.

Maintain a short evidence summary:

- searched keywords
- reusable references found
- selected API choices and why

## Non-Negotiable Workflow

Follow this sequence:

1. Parse user prompt into object/task spec
2. Read repository internals and concept-template knowledge
3. Search external references (Isaac Sim + cuRobo)
4. Send pre-coding alignment message to user
5. Wait for user confirmation
6. Implement action + test files
7. Provide Isaac manual test guide and troubleshooting

## Mandatory Pre-Coding Alignment Message

Before editing any code, send:

1. Understanding
- parsed object/task from prompt
- relevant concept templates and knowledge functions
- reusable patterns from existing tasks

2. Plan
- action file architecture
- test file state machine
- cuRobo usage plan
- evaluation metrics

3. Open questions
- unresolved assumptions only

Do not start coding until user confirms or adjusts.

## Implementation Standards

### A) Action file (`actions/franka_*_action.py`)

Build an action class that:

- follows repository lifecycle pattern: plan -> execute -> post-action
- uses deterministic seeds and deterministic planning fallback ladder
- updates world/collision states before planning
- supports task-stage decomposition (approach, interact, transition, retreat as needed)
- reuses concept-template knowledge outputs when available (`get_grasp_spec`, `articulation_state`, affordance checks, etc.)

### B) Test file (`actions/test_in_isaac_*.py`)

Build a deterministic Isaac script that:

- initializes scene and robot via repository conventions
- implements explicit finite-state execution
- prints debug-friendly logs per phase
- computes and prints task evaluation results
- terminates cleanly on success/failure

### C) Evaluation requirements

Always include:

1. Primary task metric
- metric tied to desired physical effect (distance, angle, alignment, contact success, placement error, etc.)

2. Binary success criterion
- deterministic threshold-based success decision

3. Diagnostic metrics
- extra numeric outputs helpful for tuning and failure analysis

## Task Generalization Rules

When task/object changes, keep architecture stable and swap task-specific logic only.

Examples:

- Grasp/place: position error and object final pose checks
- Pull/push: displacement along intended axis
- Open/close articulated part: joint/pose progress and final state threshold
- Rotate object part: angular displacement + axis consistency

## User Collaboration Rules

If unclear, ask immediately (do not silently assume):

- Which object instance or object id?
- Which part should be manipulated?
- What is success threshold?
- Single-step demo or multi-task sequence?

Always keep user informed before substantial code edits.

## Isaac Validation Output Requirement

Because full GUI execution may be unavailable in this environment, always provide:

1. Run procedure in Isaac Sim
- entry file and run method

2. Expected runtime behavior
- state transitions and key logs

3. Verification checklist
- what to observe visually and numerically

4. Common failures + fixes
- planning failure
- affordance/pose mismatch
- articulation not moving
- collision deadlock
- instability due to bad initial state or excessive speed

## Deliverable Checklist

Before final response, verify:

1. Generated/updated `actions/franka_*_action.py`
2. Generated/updated `actions/test_in_isaac_*.py`
3. Task metric and success criterion implemented
4. Manual Isaac test instructions included
5. Known risks and mitigations listed

## Suggested Search Keywords

- `Isaac Sim Franka manipulation example`
- `Isaac Sim articulation joint control python`
- `NVLabs curobo MotionGen plan_single`
- `curobo attach_objects_to_robot`
- `<object category> affordance manipulation Isaac`

## Primary References

- https://docs.isaacsim.omniverse.nvidia.com/
- https://github.com/NVlabs/curobo
- https://nvlabs.github.io/curobo/
- https://github.com/isaac-sim
- https://github.com/isaac-sim/IsaacLab

## local configuration
- ubuntu 22.04
- zsh
- isaac 4.0.0
- curobo v0.7.6
