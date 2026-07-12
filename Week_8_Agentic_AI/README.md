# Agentic AI Pipeline Project

```text
    +-------------------+
    |    User Query     |
    +---------+---------+
              |
    +---------v---------+
    | Single Agent Core |
    +----+----+----+----+
         |    |    |
   +-----+    |    +-----+
   |          |          |
+--v--+    +--v--+    +--v--+
|Math |    |Words|    |Other|
+-----+    +-----+    +-----+
```

## Overview

This project is a foundational implementation of a single-agent workflow designed to understand user intents, perform conditional routing, and execute specific tools. It was developed as part of Week 8: Single Agent Systems & Agent Pipelines.

The agent operates as a state processor that takes natural language queries, parses them for specific triggers, and routes the execution flow to the appropriate backend tool.

## Core Features

* **Conditional Routing:** The agent analyzes input strings and autonomously decides which tool to invoke.
* **Tool Integration:** 
  * Calculator Tool: Evaluates mathematical expressions directly from the query.
  * Keyword Extractor: Parses text and extracts words exceeding a character threshold.
* **Fallback Logic:** Unrecognized queries are safely caught and directed to a general response handler.
* **Structured Output:** All responses are normalized into a predictable JSON structure for downstream processing.

## Implementation Details

The system relies on pure Python logic rather than heavy frameworks, demonstrating the core mechanical concepts of agentic routing.

The expected JSON output structure follows this schema:
```json
{
  "type": "calculation / keywords / general / error",
  "result": "<tool_output>"
}
```

## How to Run

1. Open the Jupyter Notebook in Google Colab or your local Jupyter environment.
2. Run all cells sequentially.
3. Scroll to the bottom to find the Interactive Mode cell.
4. Type your queries to test the routing logic in real-time. Type `exit` to terminate the session.

## Example Usage

**Input:** `calculate 500 * 4`
**Output:** `{"type": "calculation", "result": "2000"}`

**Input:** `extract keywords from natural language processing algorithms`
**Output:** `{"type": "keywords", "result": ["natural", "language", "processing", "algorithms"]}`
