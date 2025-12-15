#!/usr/bin/env bun
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import z from "zod";
import { Command } from "commander";
import { Agent } from "magnitude-core";

const DEFAULT_DATASET_FILE = "patchedTasks.jsonl";
const DEFAULT_RESULTS_DIR = "results/default";

type FailureCategory = "NO_TRAJECTORY" | "NO_ANSWER" | "WRONG_EXECUTION";

interface Task {
  web_name: string;
  id: string;
  ques: string;
  web: string;
  level?: string;
}

const ERROR_ANALYSIS_PROMPT = `
You are an error analysis assistant for a web navigation benchmark.

The benchmark runs an agent in a browser to complete tasks like booking tickets or looking up information on real websites. For each task we have:
- A natural language instruction ("Instruction").
- A trajectory of observations ("Trajectory summary"), which is a step-by-step log of what the agent did.
- A previous judge model's reasoning ("Judge reasoning"), which explains why the task was marked as NOT SUCCESS.

We define three failure categories:

1. NO_TRAJECTORY:
   - There is no usable trajectory: memory.observations is missing or an empty array.
   - The run effectively produced no steps to inspect.

2. NO_ANSWER:
   - There is a trajectory with UI actions, but the agent never produced a final "answer" action.
   - The agent interacted with the page but did not return an explicit final answer.

3. WRONG_EXECUTION:
   - There is a trajectory and a final answer, but the answer does NOT fully satisfy the instruction.
   - The agent may stop too early, misread information, or ignore part of the instruction.

Your job for each failed task is:
- Decide which category best describes the failure.
- For category WRONG_EXECUTION:
  - Identify the index (0-based) of the FIRST trajectory step where the agent's action stops correctly following the instruction. If you cannot identify a specific step, use -1.
  - Provide a short explanation (2–4 sentences) of the main failure mode.

You MUST return an object with fields:
- category: one of "NO_TRAJECTORY", "NO_ANSWER", "WRONG_EXECUTION"
- failure_step: an integer index (0-based), or -1 if you cannot identify a specific step
- explanation: short natural-language explanation.
- closeness_score: a float between 0.0 and 1.0 representing how close the agent was to true success
  - 0.0 = very far from success (many essential steps missing or clearly wrong direction)
  - 0.5 = partially complete (roughly halfway, some key subgoals missing)
  - 1.0 = very close to success (most steps correct, small mistake or final detail missing)
`;

const ErrorAnalysisSchema = z.object({
  category: z.enum(["NO_TRAJECTORY", "NO_ANSWER", "WRONG_EXECUTION"]),
  failure_step: z.number().int(),
  explanation: z.string(),
  closeness_score: z.number().min(0).max(1),
});

type LLMAnalysis = z.infer<typeof ErrorAnalysisSchema>;

const NoAnswerAnalysisSchema = z.object({
  failure_step: z.number().int(),
  explanation: z.string(),
  closeness_score: z.number().min(0).max(1),
});

type NoAnswerLLMAnalysis = z.infer<typeof NoAnswerAnalysisSchema>;

interface FailureAnalysis extends LLMAnalysis {
  task_instruction: string;
  // closeness_score is inherited from LLMAnalysis (0–1 distance to success)
}

async function loadTasksMap(datasetPath: string): Promise<Map<string, Task>> {
  const tasksById = new Map<string, Task>();

  if (!fs.existsSync(datasetPath)) {
    console.warn(
      `[ErrorAnalysis] Dataset file not found at ${datasetPath}, task instructions will be missing`,
    );
    return tasksById;
  }

  const rl = readline.createInterface({
    input: fs.createReadStream(datasetPath, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const obj = JSON.parse(trimmed) as Task;
      if (obj.id) {
        tasksById.set(obj.id, obj);
      }
    } catch (e) {
      console.warn("[ErrorAnalysis] Failed to parse dataset line:", e);
    }
  }

  rl.close();
  return tasksById;
}

function extractHeuristicCategory(
  runData: any,
): {
  base: FailureCategory;
  hasMem: boolean;
  hasAnswer: boolean;
  observations: any[];
} {
  const observations = runData?.memory?.observations;
  const hasMem = Array.isArray(observations) && observations.length > 0;

  if (!hasMem) {
    return {
      base: "NO_TRAJECTORY",
      hasMem: false,
      hasAnswer: false,
      observations: [],
    };
  }

  const hasAnswer = observations.some(
    (obs: any) => obs && obs.source === "action:taken:answer",
  );

  if (!hasAnswer) {
    return {
      base: "NO_ANSWER",
      hasMem: true,
      hasAnswer: false,
      observations,
    };
  }

  return {
    base: "WRONG_EXECUTION",
    hasMem: true,
    hasAnswer: true,
    observations,
  };
}

function buildTrajectorySummary(
  observations: any[],
  maxSteps: number = 20,
): string {
  const lines: string[] = [];
  const sliced = observations.slice(0, maxSteps);

  for (let i = 0; i < sliced.length; i++) {
    const obs = sliced[i] ?? {};
    const source =
      typeof (obs as any).source === "string"
        ? (obs as any).source
        : "unknown-source";

    let actionType = "";
    const action = (obs as any).action;
    if (action && typeof action === "object") {
      if (typeof (action as any).type === "string") {
        actionType = (action as any).type;
      } else if (typeof (action as any).name === "string") {
        actionType = (action as any).name;
      }
    }

    const textFields: string[] = [];
    for (const key of Object.keys(obs)) {
      const val = (obs as any)[key];
      if (typeof val === "string") {
        if (/image|screenshot|png|jpg|jpeg|base64|data/i.test(key)) continue;
        if (val.length === 0 || val.length > 200) continue;
        textFields.push(`${key}=${val}`);
      }
    }

    const extra =
      textFields.length > 0 ? `, ${textFields.slice(0, 2).join("; ")}` : "";
    lines.push(
      `Step ${i}: source=${source}, action=${actionType || "?"}${extra}`,
    );
  }

  if (observations.length > maxSteps) {
    lines.push(
      `... (${observations.length - maxSteps} more observations omitted)`,
    );
  }

  if (lines.length === 0) {
    return "(no observation details available)";
  }

  return lines.join("\n");
}

async function analyzeFailures(opts: {
  results_dir?: string;
  dataset_file?: string;
}) {
  const resultsDir = opts.results_dir ?? DEFAULT_RESULTS_DIR;
  const datasetPath = path.join(
    __dirname,
    "data",
    opts.dataset_file ?? DEFAULT_DATASET_FILE,
  );

  const tasksMap = await loadTasksMap(datasetPath);

  if (!fs.existsSync(resultsDir)) {
    console.error(`[ErrorAnalysis] Results dir not found: ${resultsDir}`);
    process.exit(1);
  }

  const files = fs.readdirSync(resultsDir);
  const evalFiles = files.filter((f) => f.endsWith(".eval.json"));

  if (evalFiles.length === 0) {
    console.log("[ErrorAnalysis] No evaluation results found.");
    return;
  }

  const agent = new Agent({
    llm: {
      provider: "anthropic",
      options: {
        model: "claude-sonnet-4-5-20250929",
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
    },
  });
  await agent.start();

  const buckets: Record<FailureCategory, { taskIds: string[] }> = {
    NO_TRAJECTORY: { taskIds: [] },
    NO_ANSWER: { taskIds: [] },
    WRONG_EXECUTION: { taskIds: [] },
  };

  const noAnswerTimedOutIds: string[] = [];
  const details: Record<string, FailureAnalysis> = {};

  for (const evalFile of evalFiles) {
    const taskId = evalFile.replace(".eval.json", "");
    const evalPath = path.join(resultsDir, evalFile);
    const runPath = path.join(resultsDir, `${taskId}.json`);

    let evalData: any;
    try {
      evalData = JSON.parse(fs.readFileSync(evalPath, "utf-8"));
    } catch (e) {
      console.warn(
        `[ErrorAnalysis] Failed to read eval file for ${taskId}:`,
        e,
      );
      continue;
    }

    if (evalData.result !== "NOT SUCCESS") {
      continue;
    }

    const task = tasksMap.get(taskId);
    const instruction =
      task?.ques ?? "(instruction not found in dataset)";

    let runData: any = null;
    if (fs.existsSync(runPath)) {
      try {
        runData = JSON.parse(fs.readFileSync(runPath, "utf-8"));
      } catch (e) {
        console.warn(
          `[ErrorAnalysis] Failed to read run file for ${taskId}:`,
          e,
        );
      }
    }

    const { hasMem, hasAnswer, observations } =
      extractHeuristicCategory(runData ?? {});

    // NO_TRAJECTORY: purely heuristic
    if (!hasMem) {
      const explanation =
        "Run data has no memory.observations or an empty observations array.";

      const analysis: FailureAnalysis = {
        category: "NO_TRAJECTORY",
        failure_step: -1,
        explanation,
        closeness_score: 0.0,
        task_instruction: instruction,
      };

      details[taskId] = analysis;
      buckets["NO_TRAJECTORY"].taskIds.push(taskId);
      continue;
    }

    // NO_ANSWER: has trajectory but no answer -> ask LLM how close it was
    if (hasMem && !hasAnswer) {
      if (runData?.timedOut) {
        noAnswerTimedOutIds.push(taskId);
      }

      const trajSummaryNA = buildTrajectorySummary(observations);
      const judgeReasoningNA =
        typeof evalData.reasoning === "string"
          ? evalData.reasoning
          : "(no reasoning text found in eval)";

      const naPrompt = `
${ERROR_ANALYSIS_PROMPT}

This task belongs to the NO_ANSWER category:
- There is a trajectory with UI actions.
- But the agent never produced a final "answer" action.

Your job now is to analyze how close the agent was to truly completing the task, including the possibility that the UI state already satisfies the instruction but the agent simply failed to emit the final answer.

Concretely:
- If the trajectory already brought the page to a state that satisfies the instruction, explain that and say what answer the agent could have returned.
- If the trajectory is partially complete, describe what is missing and what the next key steps should have been.
- Identify the first step index (0-based) where the agent effectively stopped making real progress toward the instruction; if you cannot identify a specific step, use -1.

You MUST return:
- failure_step: integer index (0-based) or -1 if you cannot identify a specific step
- explanation: 2–4 sentences describing how close the agent was to success and what was missing.
- closeness_score: a float between 0.0 and 1.0 representing how close the agent was to true success.

Task ID: ${taskId}
Instruction:
${instruction}

Judge reasoning (for reference, may be generic or missing):
${judgeReasoningNA}

Trajectory summary (first steps):
${trajSummaryNA}
`.trim();

      try {
        const llmRes: NoAnswerLLMAnalysis = await agent.query(
          naPrompt,
          NoAnswerAnalysisSchema,
        );

        const analysis: FailureAnalysis = {
          category: "NO_ANSWER",
          failure_step: llmRes.failure_step,
          explanation: llmRes.explanation,
          closeness_score: llmRes.closeness_score,
          task_instruction: instruction,
        };

        details[taskId] = analysis;
        buckets["NO_ANSWER"].taskIds.push(taskId);
      } catch (e) {
        console.warn(
          `[ErrorAnalysis] LLM error while analyzing NO_ANSWER task ${taskId}, falling back to heuristic explanation:`,
          e,
        );

        const fallback: FailureAnalysis = {
          category: "NO_ANSWER",
          failure_step: -1,
          explanation:
            "NO_ANSWER case: trajectory has UI actions but no final answer. LLM error occurred, so we cannot estimate how close it was to success.",
          closeness_score: 0.0,
          task_instruction: instruction,
        };

        details[taskId] = fallback;
        buckets["NO_ANSWER"].taskIds.push(taskId);
      }

      continue;
    }

    // WRONG_EXECUTION: has trajectory and answer -> full error analysis
    const trajSummary = buildTrajectorySummary(observations);
    const judgeReasoning =
      typeof evalData.reasoning === "string"
        ? evalData.reasoning
        : "(no reasoning text found in eval)";

    const prompt = `
${ERROR_ANALYSIS_PROMPT}

Task ID: ${taskId}
Instruction:
${instruction}

Judge reasoning:
${judgeReasoning}

Trajectory summary (first steps):
${trajSummary}
`.trim();

    try {
      const llmAnalysis: LLMAnalysis = await agent.query(
        prompt,
        ErrorAnalysisSchema,
      );
      const analysis: FailureAnalysis = {
        ...llmAnalysis,
        task_instruction: instruction,
      };
      details[taskId] = analysis;
      buckets[llmAnalysis.category].taskIds.push(taskId);
    } catch (e) {
      console.warn(
        `[ErrorAnalysis] LLM error while analyzing ${taskId}, falling back to WRONG_EXECUTION with generic explanation:`,
        e,
      );
      const fallback: FailureAnalysis = {
        category: "WRONG_EXECUTION",
        failure_step: -1,
        explanation:
          "LLM error during error analysis; treat as WRONG_EXECUTION without a precise failing step.",
        closeness_score: 0.0,
        task_instruction: instruction,
      };
      details[taskId] = fallback;
      buckets["WRONG_EXECUTION"].taskIds.push(taskId);
    }
  }

  console.log("=== Failure Mode Summary ===");
  (["NO_TRAJECTORY", "NO_ANSWER", "WRONG_EXECUTION"] as FailureCategory[]).forEach(
    (cat) => {
      const ids = buckets[cat].taskIds;

      if (cat === "NO_ANSWER") {
        const timeoutCount = noAnswerTimedOutIds.length;
        console.log(
          `${cat}: ${ids.length} task(s) (of which ${timeoutCount} timed out)`,
        );
        if (ids.length > 0) {
          console.log(`  ids: ${ids.join(", ")}`);
        }
        if (timeoutCount > 0) {
          console.log(
            `  timedOut ids: ${noAnswerTimedOutIds.join(", ")}`,
          );
        }
      } else {
        console.log(`${cat}: ${ids.length} task(s)`);
        if (ids.length > 0) {
          console.log(`  ids: ${ids.join(", ")}`);
        }
      }
    },
  );

  const outPath = path.join(resultsDir, "failure_analysis.json");
  fs.writeFileSync(
    outPath,
    JSON.stringify(
      {
        summary: {
          NO_TRAJECTORY: {
            count: buckets.NO_TRAJECTORY.taskIds.length,
            task_ids: buckets.NO_TRAJECTORY.taskIds,
          },
          NO_ANSWER: {
            count: buckets.NO_ANSWER.taskIds.length,
            task_ids: buckets.NO_ANSWER.taskIds,
            timed_out_count: noAnswerTimedOutIds.length,
            timed_out_task_ids: noAnswerTimedOutIds,
          },
          WRONG_EXECUTION: {
            count: buckets.WRONG_EXECUTION.taskIds.length,
            task_ids: buckets.WRONG_EXECUTION.taskIds,
          },
        },
        details,
      },
      null,
      2,
    ),
  );
  console.log(
    `[ErrorAnalysis] Detailed failure analysis saved to ${outPath}`,
  );
}

const program = new Command();
program
  .name("wv-error-analysis")
  .description("Analyze failure modes for webvoyager judge results")
  .option(
    "--results_dir <dir>",
    "Directory containing *.json and *.eval.json",
    DEFAULT_RESULTS_DIR,
  )
  .option(
    "--dataset_file <file>",
    "Dataset jsonl filename under ./data (default: patchedTasks.jsonl)",
    DEFAULT_DATASET_FILE,
  )
  .action(async (options) => {
    await analyzeFailures({
      results_dir: options.results_dir,
      dataset_file: options.dataset_file,
    });
  });

program.parseAsync().catch(console.error);
