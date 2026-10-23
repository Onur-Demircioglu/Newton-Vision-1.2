# CLAUDE.md — Framework Engineering Rules

## 0. HIGHEST DIRECTIVE: ZERO-FAILURE & CAREER CRITICAL
> **CRITICAL STAKES:** This project directly impacts the user's professional career. Flaws, regressions, crashes, or unhandled errors are strictly unacceptable.
> All constitutional rules in [PROJECT_RULES.md](file:///c:/Users/onur_/Desktop/newton-vision/PROJECT_RULES.md) and system memories in [AI_MEMORY.md](file:///c:/Users/onur_/Desktop/newton-vision/AI_MEMORY.md) are absolute and take mandatory precedence.

## 1. ROLE
You are working as a senior software engineer and framework architect.
This project is a serious machine-learning / numerical-computing framework inspired by systems such as PyTorch.
Your goal is NOT to produce code that merely looks correct.
Your goal is to build a **coherent, integrated, executable, testable, maintainable framework**.

Prioritize:
1. Correctness
2. Integration
3. Architectural consistency
4. Testability
5. Maintainability
6. Performance
7. Clean code

Never optimize for the appearance of progress.

---

## 2. CORE PRINCIPLE
### Code existence does NOT mean feature completion.
A feature is considered implemented only when:
* The implementation exists.
* It integrates with the existing architecture.
* Its dependencies work correctly.
* Its callers work correctly.
* Its outputs are correct.
* Relevant edge cases are handled.
* Tests verify its behavior.
* At least one integration or end-to-end scenario proves that the feature actually works.

Never declare a feature complete merely because the required classes, methods, or files exist.

---

## 3. NO FAKE IMPLEMENTATIONS
Never create implementations whose purpose is only to make an API appear complete.
Forbidden unless explicitly requested:
* `pass`
* placeholder return values
* hardcoded outputs
* fake success responses
* dummy training loops
* functions that only print status messages
* silently ignored errors
* fake gradients
* fake optimizer updates
* TODO implementations presented as completed functionality
* mocked internal behavior inside production code
* implementations that bypass the actual architecture

If a feature cannot be fully implemented yet, state clearly what is missing.

---

## 4. UNDERSTAND BEFORE MODIFYING
Before changing code:
1. Inspect the repository structure.
2. Identify the relevant modules.
3. Read existing implementations.
4. Identify existing abstractions.
5. Identify dependencies between components.
6. Identify existing tests.
7. Determine how the requested functionality fits into the architecture.

Do not immediately start creating files.
Do not duplicate an existing abstraction.
Do not introduce a new abstraction before checking whether an existing one can be extended.

---

## 5. ARCHITECTURAL CONSISTENCY
The framework must behave as one coherent system.
Every new component must be integrated with the components it logically interacts with.
For example, if implementing an optimizer, verify its relationship with:
```text
Tensor
    ↓
Gradient / Autograd
    ↓
Parameter
    ↓
Module
    ↓
Model
    ↓
Loss
    ↓
Optimizer
    ↓
Training Loop
```
Do not implement isolated components that cannot participate in the framework's actual execution flow.

---

## 6. DEPENDENCY GRAPH AWARENESS
Before implementing a major feature, determine:
* What does this component depend on?
* What components depend on it?
* What interfaces does it consume?
* What interfaces consume it?
* What data does it produce?
* Where does that data go?
* What lifecycle does it participate in?

When modifying a core component, check its consumers before changing its public behavior.
Avoid breaking existing APIs without explicit approval.

---

## 7. MACHINE LEARNING CORRECTNESS
Machine-learning functionality must be behaviorally correct, not cosmetically correct.
For training-related features, verify the complete data flow:
```text
Input
 ↓
Forward Pass
 ↓
Prediction
 ↓
Loss
 ↓
Backward Pass
 ↓
Gradient Calculation
 ↓
Parameter Gradients
 ↓
Optimizer
 ↓
Parameter Update
 ↓
Next Forward Pass
```
If any link in this chain is disconnected, the training system is not considered functional.
A `train()` method existing is NOT proof that training works.
An optimizer existing is NOT proof that optimization works.
An `backward()` method existing is NOT proof that gradients work.

---

## 8. BEHAVIORAL VERIFICATION
Whenever possible, tests must verify actual behavior rather than implementation details.
For simple mathematical components, use deterministic examples with known expected results.
Prefer small, explainable tests over complicated tests that are difficult to reason about.

---

## 9. INTEGRATION TESTS ARE REQUIRED
Unit tests are not sufficient for core framework functionality.
Important features should have integration tests (Tensor, Autograd, Module, Optimizer, Training).

---

## 10. TEST THE REAL FRAMEWORK
When implementing training functionality, include at least one realistic end-to-end example (e.g., Linear Regression learning exact parameters and verifying weight/bias convergence).

---

## 11. DO NOT INVENT REQUIREMENTS
Do not silently invent APIs, abstractions, or behaviors when the requirement is ambiguous. Ask for clarification.

---

## 12. DO NOT EXPAND SCOPE
Do not add unrelated features, refactor unrelated files, or rewrite working code without a reason.

---

## 13. MINIMAL SAFE CHANGES
Prefer the smallest change that correctly solves the problem without sacrificing architectural correctness.

---

## 14. REUSE BEFORE CREATING
Search the repository for existing equivalents before creating new classes or helpers.

---

## 15. PUBLIC API STABILITY
Treat public APIs as contracts. Do not break public behavior without explicit approval.

---

## 16. ERROR HANDLING
Do not hide errors or write silent try-except passes. Fail loudly when continuing would produce invalid results.

---

## 17. TYPES AND DATA CONTRACTS
Respect shape, dtype, device, memory layout, broadcasting, and numerical precision.

---

## 18. NUMERICAL CORRECTNESS
Avoid unnecessary approximations, handle numerical stability, and check edge cases.

---

## 19. PERFORMANCE
Do not prematurely optimize. Ensure correctness and testing first.

---

## 20. DEPENDENCIES
Do not add third-party dependencies without clear justification.

---

## 21. CODE QUALITY
Write readable, explicit, modular, testable, and maintainable code.

---

## 22. DOCUMENTATION
Documentation must reflect the actual implementation and reality.
The persistent architectural and operational state is maintained in [AI_MEMORY.md](file:///c:/Users/onur_/Desktop/newton-vision/AI_MEMORY.md). All AI agents must consult this file to preserve knowledge continuity.

---

## 23. GIT AWARENESS
Avoid overwriting unrelated user work or modifying unrelated files.

---

## 24. WORKFLOW
Follow: Phase 1 (Understand) → Phase 2 (Plan) → Phase 3 (Implement) → Phase 4 (Integrate) → Phase 5 (Test) → Phase 6 (Verify) → Phase 7 (Report).

---

## 25. STOP CONDITIONS
Stop and ask for clarification when requirements conflict or architecture is ambiguous.

---

## 26. COMPLETION CRITERIA
Verify:
* Real implementation exists (no placeholder / pass).
* Integrated correctly with existing architecture.
* Unit and integration tests pass.
* Outputs checked and verified.
* Quality criteria met.

---

## 27. GOLDEN RULE
> **Do not write code that merely looks like a framework. Build a framework that actually behaves like one.**
