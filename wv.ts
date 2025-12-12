#!/usr/bin/env bun
import * as fs from "fs";
import * as readline from "readline";
import * as path from "path";
import z from "zod";
import { Command } from "commander";
import * as p from "@clack/prompts";
import { Agent } from "magnitude-core";
import { spawn } from "node:child_process";

const DEFAULT_DATASET_FILE = "patchedTasks.jsonl";
const DEFAULT_RESULTS_DIR = "default";

// Keep TASKS_PATH mutable for minimal changes across helpers
let TASKS_PATH = path.join(__dirname, "data", DEFAULT_DATASET_FILE);

function resolveTasksPath(datasetFile?: string) {
    const file = datasetFile || DEFAULT_DATASET_FILE;
    return path.join(__dirname, "data", file);
}

// results_dir = subfolder name under ./results
// resultsPathOpt = backward-compatible override from -r/--results-path
function resolveResultsPath(resultsDir?: string, resultsPathOpt?: string) {
    if (resultsPathOpt) return resultsPathOpt;
    const dir = resultsDir || DEFAULT_RESULTS_DIR;
    return path.join("results", dir);
}

// src: https://github.com/MinorJerry/WebVoyager/blob/main/evaluation/auto_eval.py
const EVALUATION_PROMPT = `
As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshots when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshots and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshots are authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshots, then the content of the screenshots prevails, 2) The content in the Result response is not mentioned on the screenshots, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'.
`;

interface Task {
    web_name: string;
    id: string;
    ques: string;
    web: string;

    // Online-Mind2Web / patchedTasks 里若有该字段就能读到
    level?: "easy" | "medium" | "hard" | string;
}

interface RunOptions {
    workers: string;
    eval?: boolean;
    failed?: boolean;
    failedOnly?: boolean;
    replace?: boolean;
    resultsPath?: string;   // legacy override
    dataset_file?: string;
    results_dir?: string;
}

interface EvalOptions {
    workers: string;
    replace?: boolean;
    resultsPath?: string;   // legacy override
    dataset_file?: string;
    results_dir?: string;
}

// Helper functions
async function findTaskById(
    filePath: string,
    taskId: string,
): Promise<Task | null> {
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
    });

    for await (const line of rl) {
        try {
            const task: Task = JSON.parse(line);
            if (task.id === taskId) {
                return task;
            }
        } catch (error) {
            console.error("Error parsing JSON line:", error);
        }
    }
    return null;
}

async function getAllTasks(
    filePath: string,
    category?: string,
): Promise<Task[]> {
    const tasks: Task[] = [];
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
    });

    for await (const line of rl) {
        try {
            const task: Task = JSON.parse(line);
            if (!category || task.web_name === category) {
                tasks.push(task);
            }
        } catch (error) {
            console.error("Error parsing JSON line:", error);
        }
    }
    return tasks;
}

async function getCategories(): Promise<Map<string, number>> {
    const allTasks = await getAllTasks(TASKS_PATH);
    const categories = new Map<string, number>();

    for (const task of allTasks) {
        categories.set(task.web_name, (categories.get(task.web_name) || 0) + 1);
    }

    return categories;
}

async function getTaskLevelMap(filePath: string): Promise<Map<string, string>> {
    const levelMap = new Map<string, string>();
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
    });

    for await (const line of rl) {
        try {
            const task = JSON.parse(line) as any;
            const id = task.id;
            const level = task.level ?? task.difficulty ?? "unknown";
            if (id) levelMap.set(id, String(level));
        } catch (error) {
            console.error("Error parsing JSON line for level map:", error);
        }
    }

    return levelMap;
}

function isTaskId(input: string): boolean {
    // Task IDs have format "Category--number"
    // Online-Mind2Web ids might not include "--", but we keep this logic
    // to preserve existing UX behavior.
    return input.includes("--");
}

async function selectCategories(): Promise<string[] | null> {
    const categories = await getCategories();

    // Calculate total tasks
    let totalTasks = 0;
    for (const [_, count] of categories) {
        totalTasks += count;
    }

    // First ask: all or specific
    const mode = await p.select({
        message: "Which categories would you like to run?",
        options: [
            { value: "all", label: `All Categories (${totalTasks} tasks total)` },
            { value: "specific", label: "Select specific categories" },
        ],
    });

    if (p.isCancel(mode)) {
        p.cancel("Operation cancelled");
        return null;
    }

    if (mode === "all") {
        return Array.from(categories.keys());
    }

    // User chose specific - show multiselect
    const categoryOptions = Array.from(categories.entries()).map(
        ([cat, count]) => ({
            value: cat,
            label: `${cat} (${count} tasks)`,
        }),
    );

    const selected = await p.multiselect({
        message: "Select categories:",
        options: categoryOptions,
        required: true,
    });

    if (p.isCancel(selected)) {
        p.cancel("Operation cancelled");
        return null;
    }

    return selected as string[];
}

async function selectTasksFromCategory(category: string): Promise<Task[] | null> {
    const categoryTasks = await getAllTasks(TASKS_PATH, category);

    const mode = await p.select({
        message: `Found ${categoryTasks.length} tasks in ${category}. How would you like to proceed?`,
        options: [
            { value: "all", label: `Run all ${categoryTasks.length} tasks` },
            { value: "select", label: "Select specific tasks" },
        ],
    });

    if (p.isCancel(mode)) {
        p.cancel("Operation cancelled");
        return null;
    }

    if (mode === "all") {
        return categoryTasks;
    }

    const selectedIds = await p.multiselect({
        message: "Select tasks to run:",
        options: categoryTasks.map((task) => ({
            value: task.id,
            label: `${task.id}: ${task.ques.substring(0, 80)}${task.ques.length > 80 ? "..." : ""}`,
        })),
        required: true,
    });

    if (p.isCancel(selectedIds)) {
        p.cancel("Operation cancelled");
        return null;
    }

    return categoryTasks.filter((task) =>
        (selectedIds as string[]).includes(task.id),
    );
}

async function getTaskStatus(taskId: string, resultsPath: string = "results/default"): Promise<{
    hasRun: boolean;
    hasEval: boolean;
    isSuccess: boolean;
}> {
    const resultPath = path.join(resultsPath, `${taskId}.json`);
    const evalPath = path.join(resultsPath, `${taskId}.eval.json`);

    const hasRun = fs.existsSync(resultPath);
    const hasEval = fs.existsSync(evalPath);
    let isSuccess = false;

    if (hasEval) {
        try {
            const evalData = JSON.parse(fs.readFileSync(evalPath, "utf-8"));
            isSuccess = evalData.result === "SUCCESS";
        } catch {
            // Error reading eval
        }
    }

    return { hasRun, hasEval, isSuccess };
}

type RunData = any;

function extractRunSignals(runData: RunData) {
  const actionCount = Number(runData?.actionCount ?? 0);
  const hasZeroActions = actionCount === 0;

  const hasTimedOut = runData?.timedOut === true;

  let hasAnswer = false;
  if (runData?.memory && Array.isArray(runData.memory.observations)) {
    // check whether has an answer
    hasAnswer = runData.memory.observations.some(
      (obs: any) => obs.source === "action:taken:answer",
    );
  }

  const errMsg = String(runData?.error ?? runData?.err ?? "");
  const isInitGotoTimeout =
    hasZeroActions && errMsg.includes("goto: Timeout 30000ms exceeded");

  return { actionCount, hasZeroActions, hasTimedOut, hasAnswer, isInitGotoTimeout };
}

/**
 * Default rerun / unfinished criteria:
 *  - no run file
 *  - zero actions
 *  - no answer and not timed out
 */
function isUnfinishedDefault(runExists: boolean, runData: RunData) {
  if (!runExists) return true;

  const { hasZeroActions, hasTimedOut, hasAnswer } = extractRunSignals(runData);
  return hasZeroActions || (!hasAnswer && !hasTimedOut);
}

async function filterTasksByOptions(tasks: Task[], options: RunOptions): Promise<Task[]> {
    const filteredTasks: Task[] = [];
    const resultsPath = resolveResultsPath(options.results_dir, options.resultsPath);

    for (const task of tasks) {
        const status = await getTaskStatus(task.id, resultsPath);

        if (options.replace) {
            // Run all tasks regardless of status
            filteredTasks.push(task);
        } else if (options.failedOnly) {
            // Only run failed tasks (has run but not successful)
            if (status.hasRun && !status.isSuccess) {
                filteredTasks.push(task);
            }
        } else if (options.failed) {
            // Run failed tasks and unrun tasks
            if (!status.hasRun || !status.isSuccess) {
                filteredTasks.push(task);
            }
        } else {
            const resultPath = path.join(resultsPath, `${task.id}.json`);
            const runExists = fs.existsSync(resultPath);
            let runData: any = null;

            if (runExists) {
                try {
                runData = JSON.parse(fs.readFileSync(resultPath, "utf-8"));
                } catch {
                // 读失败就当作 runData 无效
                runData = null;
                }
            }

            // 统一口径
            if (isUnfinishedDefault(runExists, runData)) {
                filteredTasks.push(task);
            }
        }
    }

    return filteredTasks;
}

// evaluate one task
async function evalTask(taskId: string, resultsPath: string = "results/default") {
    const task = (await findTaskById(TASKS_PATH, taskId))!;

    const memoryPath = path.join(resultsPath, `${task.id}.json`);
    //const memJson = JSON.parse(fs.readFileSync(memoryPath, "utf-8")).memory;
    const raw = JSON.parse(fs.readFileSync(memoryPath, "utf-8"));
    // memJson is the memory field of json file
    const memJson = raw?.memory;

    // skip the task if memory is empty or observations is not an array
    if (!memJson || !Array.isArray(memJson.observations)) {
        console.warn(`[Eval] Skip task ${task.id}: invalid memory shape`);
        return;
    }


    const agent = new Agent({
        llm: {
            // provider: 'openai-generic',
            // options: {
            //     baseUrl: 'https://openrouter.ai/api/v1',
            //     model: 'bytedance/ui-tars-1.5-7b',
            //     apiKey: process.env.OPENROUTER_API_KEY,
            // },
            // options: {
            //     baseUrl: 'https://openrouter.ai/api/v1',
            //     // model: 'qwen/qwen2.5-vl-72b-instruct',
            //     model: 'qwen/qwen3-vl-30b-a3b-instruct',
            //     apiKey: process.env.OPENROUTER_API_KEY
            // }

            provider: 'anthropic', // your provider of choice
            options: {
                // any required + optional configuration for that provider
                model: 'claude-sonnet-4-5-20250929',
                apiKey: process.env.ANTHROPIC_API_KEY,
            },
        },
    });
    await agent.start();
    // load the saved memory into the judge's memory
    await agent.memory.loadJSON(memJson);

    const evalResult = await agent.query(
        EVALUATION_PROMPT + "\n\n" + `TASK: ${task.ques}`,
        z.object({
            reasoning: z.string(),
            result: z.enum(["SUCCESS", "NOT SUCCESS"]),
        }),
    );
    console.log(evalResult);

    const evalPath = path.join(resultsPath, `${task.id}.eval.json`);
    fs.writeFileSync(evalPath, JSON.stringify(evalResult, null, 4));
}

async function runTaskAsProcess(task: Task, runEval: boolean, resultsPath: string = "results/default"): Promise<boolean> {
    return new Promise((resolve) => {
        const child = spawn('bun', [
            path.join(__dirname, 'wv-runner.ts'),
            JSON.stringify(task),
            String(runEval),
            resultsPath,
        ], {
            stdio: 'inherit',
            env: process.env,
        });

        const timeout = setTimeout(() => {
            console.error(`Process timeout for task ${task.id}, sending SIGTERM`);
            child.kill('SIGTERM');

            // ✅ give runner a short grace period to flush memory
            setTimeout(() => {
                try {
                console.error(`Force killing task ${task.id} with SIGKILL`);
                child.kill('SIGKILL');
                } catch {}
            }, 8000);

            resolve(false);
        }, 25 * 60 * 1000);// 25 minutes total timeout

        child.on('exit', (code) => {
            clearTimeout(timeout);
            if (code === 0) {
                console.log(`Process completed task ${task.id} successfully`);
                resolve(true);
            } else {
                console.error(`Process failed task ${task.id} with code ${code}`);
                resolve(false);
            }
        });

        child.on('error', (err) => {
            clearTimeout(timeout);
            console.error(`Process error for task ${task.id}:`, err);
            resolve(false);
        });
    });
}

async function runTasksParallel(tasks: Task[], workers: number, runEval: boolean = false, resultsPath: string = "results/default") {
    // Run tasks in parallel with worker processes
    let taskIndex = 0;
    let completedTasks = 0;

    const runWorker = async (workerId: number) => {
        while (taskIndex < tasks.length) {
            const currentIndex = taskIndex++;
            const task = tasks[currentIndex]!;

            console.log(
                `\n[Worker ${workerId}] Starting task ${currentIndex + 1}/${tasks.length}: ${task.id}`,
            );

            const success = await runTaskAsProcess(task, runEval, resultsPath);

            if (success) {
                completedTasks++;
                console.log(
                    `\n[Worker ${workerId}] Completed task ${currentIndex + 1}/${tasks.length}: ${task.id} (${completedTasks} total completed)`,
                );
            } else {
                completedTasks++;
                console.error(
                    `\n[Worker ${workerId}] Failed task ${currentIndex + 1}/${tasks.length}: ${task.id}`,
                );
            }
        }
    };

    const workerPromises: Promise<void>[] = [];
    for (let i = 0; i < workers; i++) {
        workerPromises.push(runWorker(i + 1));
    }

    await Promise.all(workerPromises);

    console.log(`\nAttempted ${tasks.length} task${tasks.length !== 1 ? "s" : ""}`);
}

async function evalTasksParallel(taskIds: string[], workers: number, resultsPath: string = "results/default") {
    let taskIndex = 0;
    let completedTasks = 0;

    const runWorker = async (workerId: number) => {
        while (taskIndex < taskIds.length) {
            const currentIndex = taskIndex++;
            const taskId = taskIds[currentIndex]!;

            console.log(
                `\n[Worker ${workerId}] Starting evaluation ${currentIndex + 1}/${taskIds.length}: ${taskId}`,
            );

            try {
                await evalTask(taskId, resultsPath);
                completedTasks++;
                console.log(
                    `\n[Worker ${workerId}] Completed evaluation ${currentIndex + 1}/${taskIds.length}: ${taskId} (${completedTasks} total completed)`,
                );
            } catch (error) {
                console.error(
                    `\n[Worker ${workerId}] Error evaluating task ${taskId}:`,
                    error,
                );
                completedTasks++;
            }
        }
    };

    const workerPromises: Promise<void>[] = [];
    for (let i = 0; i < workers; i++) {
        workerPromises.push(runWorker(i + 1));
    }

    await Promise.all(workerPromises);

    console.log(`\nCompleted evaluation of ${taskIds.length} task${taskIds.length !== 1 ? "s" : ""}`);
}

// Commands
const program = new Command();

program
    .command("run [input]")
    .description("Run tasks by category or task ID")
    .option("-w, --workers <number>", "Number of parallel workers", "1")
    .option("--eval", "Automatically run evaluation after each task")
    .option("--failed", "Include failed tasks (default: only incomplete tasks) - useful for pass@N")
    .option("--failed-only", "Only run failed tasks")
    .option("--replace", "Run all tasks regardless of status")
    .option("-r, --results-path <path>", "Path to results directory")
    .option("--dataset_file <name>", "Dataset file under ./data", DEFAULT_DATASET_FILE)
    .option("--results_dir <name>", "Subfolder under ./results", DEFAULT_RESULTS_DIR)
    .action(async (input: string | undefined, options: RunOptions) => {
        // Resolve paths from new params (and keep -r as override)
        TASKS_PATH = resolveTasksPath(options.dataset_file);
        const resultsPath = resolveResultsPath(options.results_dir, options.resultsPath);

        const workers = parseInt(options.workers);
        let tasksToRun: Task[] = [];

        if (input && isTaskId(input)) {
            // Single task ID provided
            const task = await findTaskById(TASKS_PATH, input);
            if (!task) {
                console.error(`Task ${input} not found`);
                return;
            }
            tasksToRun = [task];
        } else if (input) {
            // Category name provided
            const categoryTasks = await getAllTasks(TASKS_PATH, input);
            if (categoryTasks.length === 0) {
                console.error(`No tasks found for category: ${input}`);
                return;
            }

            // Ask for task selection
            const selectedTasks = await selectTasksFromCategory(input);
            if (!selectedTasks) return;

            tasksToRun = await filterTasksByOptions(selectedTasks, options);
        } else {
            // No input - ask for categories
            const selectedCategories = await selectCategories();
            if (!selectedCategories) return;

            if (selectedCategories.length === 1) {
                // Single category - ask for task selection
                const selectedTasks = await selectTasksFromCategory(selectedCategories[0]!);
                if (!selectedTasks) return;

                tasksToRun = await filterTasksByOptions(selectedTasks, options);
            } else {
                // Multiple categories - run all tasks in each
                for (const category of selectedCategories) {
                    const categoryTasks = await getAllTasks(TASKS_PATH, category);
                    const filteredTasks = await filterTasksByOptions(categoryTasks, options);
                    tasksToRun.push(...filteredTasks);
                }
            }
        }

        if (tasksToRun.length === 0) {
            console.log("No tasks match the criteria");
            return;
        }

        // Ensure results directory exists
        if (!fs.existsSync(resultsPath)) {
            fs.mkdirSync(resultsPath, { recursive: true });
        }

        p.outro(`Running ${tasksToRun.length} task${tasksToRun.length !== 1 ? "s" : ""} with ${workers} worker${workers !== 1 ? "s" : ""}`);

        await runTasksParallel(tasksToRun, workers, options.eval || false, resultsPath);
    });

program
    .command("eval [input]")
    .description("Evaluate tasks by category or task ID")
    .option("-w, --workers <number>", "Number of parallel workers", "1")
    .option("--replace", "Re-run evaluations even if they already exist")
    .option("-r, --results-path <path>", "Path to results directory")
    .option("--dataset_file <name>", "Dataset file under ./data", DEFAULT_DATASET_FILE)
    .option("--results_dir <name>", "Subfolder under ./results", DEFAULT_RESULTS_DIR)
    .action(async (input: string | undefined, options: EvalOptions) => {
        // Resolve paths from new params (and keep -r as override)
        TASKS_PATH = resolveTasksPath(options.dataset_file);
        const resultsPath = resolveResultsPath(options.results_dir, options.resultsPath);

        const workers = parseInt(options.workers);
        let taskIdsToEval: string[] = [];

        if (input && isTaskId(input)) {
            // Single task ID provided
            taskIdsToEval = [input];
        } else if (input) {
            // Category name provided
            const categoryTasks = await getAllTasks(TASKS_PATH, input);
            if (categoryTasks.length === 0) {
                console.error(`No tasks found for category: ${input}`);
                return;
            }

            // Filter to tasks that have been run
            for (const task of categoryTasks) {
                const status = await getTaskStatus(task.id, resultsPath);
                if (status.hasRun && (options.replace || !status.hasEval)) {
                    taskIdsToEval.push(task.id);
                }
            }
        } else {
            // No input - ask for categories
            const selectedCategories = await selectCategories();
            if (!selectedCategories) return;

            for (const category of selectedCategories) {
                const categoryTasks = await getAllTasks(TASKS_PATH, category);
                for (const task of categoryTasks) {
                    const status = await getTaskStatus(task.id, resultsPath);
                    if (status.hasRun && (options.replace || !status.hasEval)) {
                        taskIdsToEval.push(task.id);
                    }
                }
            }
        }

        if (taskIdsToEval.length === 0) {
            console.log("No tasks need evaluation");
            return;
        }

        // Ensure results directory exists
        if (!fs.existsSync(resultsPath)) {
            fs.mkdirSync(resultsPath, { recursive: true });
        }

        p.outro(`Evaluating ${taskIdsToEval.length} task${taskIdsToEval.length !== 1 ? "s" : ""} with ${workers} worker${workers !== 1 ? "s" : ""}`);

        await evalTasksParallel(taskIdsToEval, workers, resultsPath);
    });

program
    .command("stats")
    .description("Show evaluation statistics")
    .option("-v, --verbose", "Show detailed stats for each task")
    .option("-r, --results-path <path>", "Path to results directory")
    .option("--dataset_file <name>", "Dataset file under ./data", DEFAULT_DATASET_FILE)
    .option("--results_dir <name>", "Subfolder under ./results", DEFAULT_RESULTS_DIR)
    .action(async (options: any) => {
        TASKS_PATH = resolveTasksPath(options.dataset_file);
        const resultsPath = resolveResultsPath(options.results_dir, options.resultsPath);
        await showStats(options.verbose || false, resultsPath);
    });

function summarize(arr: number[]) {
    const n = arr.length;
    if (n === 0) {
        return { n: 0, min: 0, max: 0, mean: 0, median: 0 };
    }
    const sorted = [...arr].sort((a, b) => a - b);
    const sum = arr.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    const median =
        n % 2 === 1
            ? sorted[(n - 1) / 2]!
            : (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2;

    return {
        n,
        min: sorted[0]!,
        max: sorted[n - 1]!,
        mean,
        median,
    };
}

async function showStats(
  verbose: boolean = false,
  resultsPath: string = "results/default",
) {
  const resultsDir = resultsPath;
  if (!fs.existsSync(resultsDir)) {
    console.log(`No results directory found at: ${resultsDir}`);
    return;
  }

  const files = fs.readdirSync(resultsDir);
  const evalFiles = files.filter((f) => f.endsWith(".eval.json"));

  if (evalFiles.length === 0) {
    console.log("No evaluation results found.");
    return;
  }

  const levelMap = await getTaskLevelMap(TASKS_PATH);

  const levelStats = new Map<string, { total: number; success: number }>();
  for (const lv of ["easy", "medium", "hard", "unknown"]) {
    levelStats.set(lv, { total: 0, success: 0 });
  }

  const categoryStats = new Map<
    string,
    {
      total: number;
      success: number;

      // Avg actions only computed from tasks that have id.json
      totalActions: number;
      actionsCounted: number;

      actionCounts: number[];
      timesMs: number[];
      inputTokens: number[];
      outputTokens: number[];

      tasks?: Array<{
        taskId: string;
        success: boolean;
        actions?: number;
        timeMs?: number;
        inputTokens?: number;
        outputTokens?: number;
      }>;
    }
  >();

  // overall arrays (only from id.json)
  const allActionCounts: number[] = [];
  const allTimesMs: number[] = [];
  const allInputTokens: number[] = [];
  const allOutputTokens: number[] = [];

  let tasksWithRunStats = 0;

  // Unfinished breakdown counters (mutually exclusive, default rerun criteria)
  let unfinishedTasks = 0;

  let noRunFileTasks = 0;         // missing run stats json
  let initGotoTimeoutTasks = 0;   // zero actions + init goto timeout error
  let zeroActionsOtherTasks = 0;  // zero actions but not init goto timeout
  let noAnswerNoTimeoutTasks = 0; // no answer + not timed out

  // Tracked info (NOT part of default unfinished)
  let timedOutTasks = 0;

  for (const evalFile of evalFiles) {
    const taskId = evalFile.replace(".eval.json", "");
    const evalPath = path.join(resultsDir, evalFile);
    const runPath = path.join(resultsDir, `${taskId}.json`);

    try {
      const evalData = JSON.parse(fs.readFileSync(evalPath, "utf-8"));
      const isSuccess = evalData.result === "SUCCESS";

      const category = taskId.includes("--")
        ? taskId.split("--")[0]!
        : "online_mind2web";

      const level = levelMap.get(taskId) ?? "unknown";
      if (!levelStats.has(level)) {
        levelStats.set(level, { total: 0, success: 0 });
      }

      const ls = levelStats.get(level)!;
      ls.total += 1;
      if (isSuccess) ls.success += 1;

      if (!categoryStats.has(category)) {
        categoryStats.set(category, {
          total: 0,
          success: 0,
          totalActions: 0,
          actionsCounted: 0,
          actionCounts: [],
          timesMs: [],
          inputTokens: [],
          outputTokens: [],
          tasks: verbose ? [] : undefined,
        });
      }

      const stats = categoryStats.get(category)!;
      stats.total += 1;
      if (isSuccess) stats.success += 1;

      const runExists = fs.existsSync(runPath);
      let runData: any = null;

      if (runExists) {
        runData = JSON.parse(fs.readFileSync(runPath, "utf-8"));
      }

    if (runData) {
        const sig = extractRunSignals(runData);

        const taskActions = sig.actionCount;
        const taskTimeMs = Number(runData.time ?? 0);
        const taskInTokens = Number(runData.totalInputTokens ?? 0);
        const taskOutTokens = Number(runData.totalOutputTokens ?? 0);

        // ===== timed out 作为信息项单独统计 =====
        if (sig.hasTimedOut) {
            timedOutTasks += 1;
        }

        // ===== default unfinished: mutually exclusive reasons =====
        // 口径与默认重跑一致：
        //   - zero actions
        //   - or no answer AND not timed out
        if (sig.hasZeroActions) {
            if (sig.isInitGotoTimeout) {
            initGotoTimeoutTasks += 1;
            } else {
            zeroActionsOtherTasks += 1;
            }
            unfinishedTasks += 1;
        } else if (!sig.hasAnswer && !sig.hasTimedOut) {
            noAnswerNoTimeoutTasks += 1;
            unfinishedTasks += 1;
        }

        // ===== 你之前丢掉的聚合逻辑放这里 =====
        tasksWithRunStats += 1;

        stats.totalActions += taskActions;
        stats.actionsCounted += 1;

        stats.actionCounts.push(taskActions);
        stats.timesMs.push(taskTimeMs);
        stats.inputTokens.push(taskInTokens);
        stats.outputTokens.push(taskOutTokens);

        allActionCounts.push(taskActions);
        allTimesMs.push(taskTimeMs);
        allInputTokens.push(taskInTokens);
        allOutputTokens.push(taskOutTokens);

        if (verbose && stats.tasks) {
            stats.tasks.push({
            taskId,
            success: isSuccess,
            actions: taskActions,
            timeMs: taskTimeMs,
            inputTokens: taskInTokens,
            outputTokens: taskOutTokens,
            });
        }
    } else {
        // eval exists but run stats json missing
        noRunFileTasks += 1;
        unfinishedTasks += 1;

        if (verbose && stats.tasks) {
            stats.tasks.push({
            taskId,
            success: isSuccess,
            });
        }
    }
    } catch (error) {
      console.error(`Error processing ${evalFile}:`, error);
    }
  }

  // ===== Category table =====
  console.log("\n=== Evaluation Statistics by Category ===\n");
  console.log("Category         | Success Rate      | Avg Actions");
  console.log("-----------------|-------------------|------------");

  let totalTasks = 0;
  let totalSuccess = 0;

  for (const [category, stats] of categoryStats) {
    const successRate = (stats.success / stats.total) * 100;
    const avgActions =
      stats.actionsCounted > 0
        ? stats.totalActions / stats.actionsCounted
        : 0;

    console.log(
      `${category.padEnd(16)} | ${stats.success}/${stats.total} (${successRate.toFixed(1)}%)`.padEnd(
        37,
      ) + ` | ${avgActions.toFixed(1).padStart(10)}`,
    );

    if (verbose && stats.tasks) {
      stats.tasks.sort((a, b) => a.taskId.localeCompare(b.taskId));
      for (const task of stats.tasks) {
        const status = task.success ? "✓" : "✗";
        if (task.actions == null) {
          console.log(`  ${status} ${task.taskId.padEnd(20)} | (no run stats json)`);
        } else {
          const timeMin = ((task.timeMs ?? 0) / 1000 / 60).toFixed(1);
          console.log(
            `  ${status} ${task.taskId.padEnd(20)} | Actions: ${task.actions
              .toString()
              .padStart(3)} | Time: ${timeMin.padStart(5)} min | InTok: ${(task.inputTokens ?? 0)
              .toString()
              .padStart(6)} | OutTok: ${(task.outputTokens ?? 0).toString().padStart(6)}`,
          );
        }
      }
      console.log();
    }

    totalTasks += stats.total;
    totalSuccess += stats.success;
  }

  console.log("-----------------|-------------------|------------");
  const overallSuccessRate = (totalSuccess / totalTasks) * 100;

  // overall avg actions only over tasks with run stats
  const overallAvgActions =
    tasksWithRunStats > 0
      ? allActionCounts.reduce((a, b) => a + b, 0) / tasksWithRunStats
      : 0;

  console.log(
    `${"TOTAL".padEnd(16)} | ${totalSuccess}/${totalTasks} (${overallSuccessRate.toFixed(1)}%)`.padEnd(
      37,
    ) + ` | ${overallAvgActions.toFixed(1).padStart(10)}`,
  );

  console.log("\n=== Accuracy by Level ===\n");

  const preferredOrder = ["easy", "medium", "hard", "unknown"];
  for (const lv of preferredOrder) {
    const s = levelStats.get(lv);
    if (!s || s.total === 0) continue;
    const rate = (s.success / s.total) * 100;
    console.log(`${lv.padEnd(8)} | ${s.success}/${s.total} (${rate.toFixed(1)}%)`);
  }

  // 如果你想把所有“非标准 level”也打出来：
  for (const [lv, s] of levelStats) {
    if (preferredOrder.includes(lv)) continue;
    if (s.total === 0) continue;
    const rate = (s.success / s.total) * 100;
    console.log(`${lv.padEnd(8)} | ${s.success}/${s.total} (${rate.toFixed(1)}%)`);
  }

  // ===== New overall metric statistics =====
  const timeMinutes = allTimesMs.map((ms) => ms / 1000 / 60);

  const timeStats = summarize(timeMinutes);
  const actionStats = summarize(allActionCounts);
  const inTokStats = summarize(allInputTokens);
  const outTokStats = summarize(allOutputTokens);

  console.log("\n=== Overall Task Metric Statistics ===\n");
  console.log(`Evaluated tasks: ${totalTasks}`);
  console.log(`Tasks with run stats json: ${tasksWithRunStats}/${totalTasks}\n`);

  // Unfinished summary first (as requested)
  console.log(`Unfinished tasks (default rerun criteria): ${unfinishedTasks}/${totalTasks}`);

  // Unfinished breakdown (some categories may overlap)
  console.log(`  - No run stats json: ${noRunFileTasks}/${totalTasks}`);
  console.log(`  - Init goto timeout (zero actions): ${initGotoTimeoutTasks}/${totalTasks}`);
  console.log(`  - Zero actions (other): ${zeroActionsOtherTasks}/${totalTasks}`);
  console.log(`  - No answer (has actions, not timed out): ${noAnswerNoTimeoutTasks}/${totalTasks}`);
  console.log(`Timed out tasks: ${timedOutTasks}/${totalTasks}\n`);

  console.log("Metric                |   Min   |   Max   |   Mean  |  Median");
  console.log("----------------------|---------|---------|---------|---------");
  console.log(
    `Time (min)            | ${timeStats.min.toFixed(1).padStart(7)} | ${timeStats.max
      .toFixed(1)
      .padStart(7)} | ${timeStats.mean.toFixed(1).padStart(7)} | ${timeStats.median
      .toFixed(1)
      .padStart(7)}`,
  );
  console.log(
    `Action count          | ${actionStats.min.toFixed(1).padStart(7)} | ${actionStats.max
      .toFixed(1)
      .padStart(7)} | ${actionStats.mean.toFixed(1).padStart(7)} | ${actionStats.median
      .toFixed(1)
      .padStart(7)}`,
  );
  console.log(
    `Total input tokens    | ${inTokStats.min.toFixed(1).padStart(7)} | ${inTokStats.max
      .toFixed(1)
      .padStart(7)} | ${inTokStats.mean.toFixed(1).padStart(7)} | ${inTokStats.median
      .toFixed(1)
      .padStart(7)}`,
  );
  console.log(
    `Total output tokens   | ${outTokStats.min.toFixed(1).padStart(7)} | ${outTokStats.max
      .toFixed(1)
      .padStart(7)} | ${outTokStats.mean.toFixed(1).padStart(7)} | ${outTokStats.median
      .toFixed(1)
      .padStart(7)}`,
  );
}


program.parseAsync().catch(console.error);
