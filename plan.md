# LU10 AI Capstone Project: Architecture & Implementation Plan

## 1. Chosen Problem
University and college students frequently navigate fragmented, lengthy, and dense academic documents across departmental websites, learning management systems (Canvas/Blackboard), syllabus PDFs, and registrar pages. When seeking critical information—such as course prerequisites, module progression, grading thresholds, late submission penalties, academic integrity policies, advising hours, or tutoring resources—students struggle with information silos, contradictory guidance, and missed academic deadlines. 

Furthermore, students regularly face planning bottlenecks: calculating exact timelines until impending assignment deadlines and realistically apportioning limited study hours across complex academic subjects.

The **AI Campus Knowledge Assistant** solves this problem by providing a centralized, interactive, multi-turn AI assistant that grounds factual answers in verified campus documentation, performs deterministic academic calculations, safeguards against hallucinations, and refuses out-of-scope or adversarial requests.

---

## 2. Target User
- **Primary Audience:** College and university undergraduate and graduate students.
- **Academic Context:** Enrolled in intensive technical and interdisciplinary curricula (e.g., Computer Science, Machine Learning, Data Science).
- **Primary User Goals:**
  1. Retrieve authoritative course and university policy answers rapidly without manual document searching.
  2. Maintain context across multi-turn inquiries (e.g., asking about a deadline and then immediately asking "How many days do I have left?").
  3. Calculate days remaining until academic milestones and schedule study hours across multiple topics.
  4. Ensure zero hallucination: receive transparent source citations or an explicit fallback message when information is unavailable.

---

## 3. The Three LangGraph Routes

The system employs a cyclical, state-driven LangGraph workflow featuring shared typed state, conversational question disambiguation, and conditional branching:

```
[User Question]
       │
       ▼
[rewrite_node] ◄── Resolves conversational pronouns & history references
       │
       ▼
[router_node]  ◄── Evaluates intent, safety, and adversarial patterns
       │
   ┌───┼────────────────────────────────────────┐
   │   ▼                                        ▼
[Route A: DOCS]                          [Route B: TOOLS]                [Route C: DECLINE]
   │                                        │                                   │
   ├── docs_retrieval_node                  └── tools_agent_node                └── decline_node
   │   (Chroma vector similarity)               (Input extraction)                  (Polite refusal)
   ├── docs_context_node                        └── tool_execution_node                 │
   │   (Format chunks & sources)                    (Python tool calculation)           ▼
   └── docs_llm_node                                │                                 [END]
       (Gemini grounded generation)                 ▼
   │                                              [END]
   ▼
 [END]
```

### Route A: DOCS (Knowledge Base Ingestion & Grounded Retrieval)
- **Purpose:** Answers student questions regarding course syllabi, module breakdowns, prerequisites, grading rubrics, attendance requirements, academic integrity guidelines, advising hours, and campus support facilities.
- **Workflow Pipeline:**
  `Question → rewrite_node → router_node → docs_retrieval_node → docs_context_node → docs_llm_node → Cited Grounded Answer`.
- **Grounding & Fallback:** Strictly constrained to retrieved chunks. If relevant context is absent or similarity distance exceeds threshold, returns the exact standardized fallback:
  `"I couldn't find that information in the available documents."`

### Route B: TOOLS (Deterministic Python Academic Calculations)
- **Purpose:** Executes deterministic calculations and planning tasks without external side-effects.
- **Workflow Pipeline:**
  `Question → rewrite_node → router_node → tools_agent_node → tool_execution_node → Structured Output`.
- **Execution Logging:** Records tool parameters, inputs, and outputs in an expandable `Agent Steps` inspection container.

### Route C: DECLINE (Safety Boundaries & Scope Guard)
- **Purpose:** Gracefully intercepts and refuses out-of-scope requests (e.g. food recipes, entertainment, sports, cryptocurrency, general trivia) as well as adversarial jailbreaks and prompt injection attempts.
- **Workflow Pipeline:**
  `Question → rewrite_node → router_node → decline_node → Polite Refusal`.
- **Standard Refusal Message:**
  `"I'm designed to assist with campus and academic information. I can't help with that request."`

---

## 4. The Three Knowledge Documents

Stored within the `knowledge_base/` directory, these documents provide mutually consistent, comprehensive campus information:

1. **`course_overview.md` (CS 401 Course Overview & Syllabus):**
   - Details course description, instructors (Dr. Elena Vance), teaching assistants (Marcus Brody, Priya Sharma), meeting times, and lab locations.
   - Prerequisites: CS 201 (minimum grade B / 3.0 GPA) and MATH 215 (minimum grade C+ / 2.3 GPA).
   - Five curriculum modules:
     - Module 1: Foundations of Machine Learning & Embeddings.
     - Module 2: Retrieval-Augmented Generation (RAG) Architecture.
     - Module 3: Autonomous Agents & Graph Workflows.
     - Module 4: Conversation Memory & Context Management.
     - Module 5: AI Safety, Evaluation & Production Deployment.
   - Assessment dates: Assignment 1 (Feb 15, 2027), Midterm Exam (Mar 11, 2027), Assignment 2 (Apr 5, 2027), Final Capstone Project (May 10, 2027).

2. **`academic_policies.md` (Campus Academic Policies & Regulations):**
   - University standardized grading scale (A through F thresholds and GPA weights).
   - Late submission penalties: 10% deduction per 24 hours up to 3 days (30% maximum); zero score after 72 hours; emergency extension requirements (12 hours advance petition); final capstones require Academic Dean approval.
   - Academic integrity and AI usage guidelines: mandatory documentation of AI assistance, zero score and Honor Board reporting for unattributed generation.
   - Attendance requirements: minimum 80% attendance required for course credit.
   - Add/drop deadline (10 instructional days) and course withdrawal deadline (April 2, 2027).

3. **`student_guide.md` (Student Guide & Academic Support Services):**
   - Departmental and faculty advising: locations (Student Services Center Suite 210), office hours, walk-in schedules.
   - Free tutoring resources: STEM Learning Hub (Turing Hall 104) and University Writing Lab (Library Annex 3B).
   - Weekly instructor and TA office hours schedules and locations.
   - 24/7 Turing Computing Labs, HPC cluster access requests, and $50 student Google Cloud/Gemini API credits.
   - Mental health counseling (Health Center 2nd floor, 24/7 crisis hotline) and Accessibility & Disability Services (ADS).

---

## 5. The Two Custom Tools

Implemented in `tools.py` with strict Python type hints, single-sentence docstrings, input validation, and zero external side effects:

### Tool 1: Academic Deadline Calculator (`days_until_deadline`)
- **Signature:** `days_until_deadline(deadline: str) -> str`
- **Docstring:** `"Calculate the number of days remaining until an academic deadline."`
- **Capabilities:**
  - Robust flexible date parsing across diverse formats (`YYYY-MM-DD`, `Month DD, YYYY`, `DD/MM/YYYY`, etc.).
  - Distinguishes future deadlines (`X days remaining`), tomorrow (`Tomorrow, 1 day remaining`), today (`is TODAY!`), and past deadlines (`passed X days ago`).
  - Graceful validation error messaging on malformed date inputs.

### Tool 2: Study Schedule Calculator (`calculate_study_schedule`)
- **Signature:** `calculate_study_schedule(topics: str, available_hours: float) -> str`
- **Docstring:** `"Distribute available study hours evenly across requested academic topics."`
- **Capabilities:**
  - Accepts comma-separated topic lists and positive total hours.
  - Automatically deduplicates and cleans topic lists.
  - Formats time allocations intelligently (in hours or minutes depending on duration).
  - Supplies evidence-based study guidance (Pomodoro 25/5 rhythm).
  - Rigorous input validation rejecting negative, zero, or non-numeric hour allocations.

---

## 6. The Extra Feature: Interactive Suggested Questions
To eliminate the "blank chat input" barrier for college students arriving at the application:
- Four interactive example question buttons are prominently displayed on the homepage when conversation history is empty:
  1. `📖 What topics are covered in the course?` (Exercises Route A / Course Overview)
  2. `⚖️ What are the academic policies?` (Exercises Route A / Academic Policies)
  3. `⏳ How many days remain until my deadline?` (Exercises Route B / Deadline Tool)
  4. `📅 Help me create a study schedule.` (Exercises Route B / Study Schedule Tool)
- Clicking any suggested question triggers the complete LangGraph workflow without hardcoded responses, demonstrating live routing, execution path logging, and response generation.
