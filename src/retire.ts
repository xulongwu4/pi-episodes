import type { AgentMessage } from '@earendil-works/pi-agent-core';
import type { AssistantMessage } from '@earendil-works/pi-ai';
import type { ProjectedSessionEntry, SessionEntry } from '@earendil-works/pi-coding-agent';

function retireTextSignature(signature: string | undefined): string | undefined {
  if (!signature) return undefined;
  if (signature.startsWith('{')) {
    try {
      const parsed = JSON.parse(signature) as {
        v?: unknown;
        id?: unknown;
        phase?: unknown;
      };
      if (parsed.v === 1 && typeof parsed.id === 'string') {
        if (parsed.phase === 'commentary' || parsed.phase === 'final_answer') {
          return JSON.stringify({ v: 1, id: '', phase: parsed.phase });
        }
        return undefined;
      }
    } catch {
      return undefined;
    }
  }
  return undefined;
}

/**
 * Retires provider replay signatures and encrypted/redacted reasoning on an assistant message's
 * content once an edit before it changed the transcript prefix. Tool call ids stay untouched so
 * they still match `ToolResultMessage.toolCallId`; stale Responses item ids are stripped on the
 * wire payload instead. Returns undefined when nothing in `content` needed retiring.
 */
export function retireAssistantContent(
  content: AssistantMessage['content']
): AssistantMessage['content'] | undefined {
  let changed = false;
  const next: AssistantMessage['content'] = [];
  for (const block of content) {
    if (block.type === 'thinking') {
      if (block.redacted) {
        changed = true;
        continue;
      }
      if (!block.thinking || block.thinking.trim() === '') {
        changed = true;
        continue;
      }
      // OpenAI Chat Completions stores the plain reasoning field key ('reasoning_content',
      // 'reasoning', 'reasoning_text') in `thinkingSignature` for DeepSeek, Kimi, Qwen, ZAI, and
      // llama.cpp — keep those field selectors intact while stripping cryptographic signatures.
      if (
        block.thinkingSignature !== undefined &&
        block.thinkingSignature !== 'reasoning_content' &&
        block.thinkingSignature !== 'reasoning' &&
        block.thinkingSignature !== 'reasoning_text'
      ) {
        changed = true;
        const { thinkingSignature: _sig, redacted: _red, ...rest } = block;
        next.push(rest);
      } else {
        next.push(block);
      }
      continue;
    }
    if (block.type === 'text') {
      const textSignature = retireTextSignature(block.textSignature);
      if (textSignature !== block.textSignature) {
        changed = true;
        const { textSignature: _old, ...rest } = block;
        next.push(textSignature === undefined ? rest : { ...rest, textSignature });
      } else {
        next.push(block);
      }
      continue;
    }
    if (block.type === 'toolCall') {
      if (block.thoughtSignature !== undefined) {
        changed = true;
        const { thoughtSignature: _sig, ...rest } = block;
        next.push(rest);
      } else {
        next.push(block);
      }
      continue;
    }
    next.push(block);
  }
  const hasBody = next.some(
    (block) => block.type === 'toolCall' || (block.type === 'text' && block.text.trim().length > 0)
  );
  if (!hasBody) {
    const thoughts = next.filter(
      (block): block is Extract<AssistantMessage['content'][number], { type: 'thinking' }> =>
        block.type === 'thinking' && block.thinking.trim().length > 0
    );
    if (thoughts.length > 0) {
      return thoughts.map((block) => ({ type: 'text' as const, text: block.thinking }));
    }
    return changed ? [{ type: 'text', text: '[thinking retired]' }] : undefined;
  }
  return changed ? next : undefined;
}

/** Returns `message` with replay signatures retired, or the original reference when unchanged. */
export function retireAssistantMessage(message: AssistantMessage): AssistantMessage {
  const content = retireAssistantContent(message.content);
  return content ? { ...message, content } : message;
}

/**
 * Responses `call_id` values from `messages` starting at `fromIndex` (default 0) whose assistant
 * turn no longer carries a signed Responses reasoning item (`rs_*`): the part before `|` on each
 * `toolCall` block. Turns that still carry a signed `thinking` block keep their `(rs_*, fc_*)`
 * pair intact.
 */
export function retiredResponsesCallIds(
  messages: readonly AgentMessage[],
  fromIndex = 0
): Set<string> {
  const ids = new Set<string>();
  for (let i = Math.max(0, fromIndex); i < messages.length; i++) {
    const message = messages[i]!;
    if (message.role !== 'assistant') continue;
    const stillSignedResponses = message.content.some(
      (block) =>
        block.type === 'thinking' &&
        typeof block.thinkingSignature === 'string' &&
        block.thinkingSignature.startsWith('{')
    );
    if (stillSignedResponses) continue;
    for (const block of message.content) {
      if (block.type !== 'toolCall') continue;
      const callId = block.id.split('|')[0];
      if (callId) ids.add(callId);
    }
  }
  return ids;
}

/**
 * Whether `branch` has any `context_edit` targeting an entry in `projection`.
 */
export function branchHasMutatedHistory(
  branch: readonly SessionEntry[],
  projection: readonly ProjectedSessionEntry[]
): boolean {
  const editedTargets = new Set<string>();
  for (const entry of branch) {
    if (entry.type === 'context_edit') editedTargets.add(entry.targetId);
  }
  if (editedTargets.size === 0) return false;
  return projection.some((entry) => editedTargets.has(entry.sourceEntry.id));
}

/**
 * Responses `call_id` values for assistant turns at or after the earliest `context_edit` target
 * on `branch` in `projection` and at or before the latest `context_edit` on `branch`. Turns
 * created after the latest edit keep their item ids untouched for incremental continuation.
 */
export function branchRetiredResponsesCallIds(
  branch: readonly SessionEntry[],
  projection: readonly ProjectedSessionEntry[]
): Set<string> {
  const branchIndexById = new Map<string, number>();
  const editedTargets = new Set<string>();
  let lastEditBranchIndex = -1;
  for (let index = 0; index < branch.length; index++) {
    const entry = branch[index]!;
    branchIndexById.set(entry.id, index);
    if (entry.type === 'context_edit') {
      editedTargets.add(entry.targetId);
      lastEditBranchIndex = index;
    }
  }
  if (editedTargets.size === 0) return new Set();
  const firstMutated = projection.findIndex((entry) => editedTargets.has(entry.sourceEntry.id));
  if (firstMutated < 0) return new Set();
  const tail: AgentMessage[] = [];
  for (let i = firstMutated; i < projection.length; i++) {
    const projected = projection[i]!;
    const branchIdx = branchIndexById.get(projected.sourceEntry.id) ?? Number.POSITIVE_INFINITY;
    if (branchIdx > lastEditBranchIndex) continue;
    tail.push(...projected.messages);
  }
  return retiredResponsesCallIds(tail);
}

/**
 * Strips `id` (`fc_*` / `ctc_*`) from OpenAI / Codex / Azure Responses `function_call` and
 * `custom_tool_call` input items whose `call_id` is in `retiredCallIds`. Returns the original
 * `payload` reference when nothing changed.
 */
export function stripRetiredResponsesIds(
  payload: unknown,
  retiredCallIds: ReadonlySet<string>
): unknown {
  if (retiredCallIds.size === 0 || !payload || typeof payload !== 'object') return payload;
  const record = payload as { input?: unknown };
  if (!Array.isArray(record.input)) return payload;
  let changed = false;
  const input = record.input.map((item: unknown) => {
    if (!item || typeof item !== 'object') return item;
    const wire = item as Record<string, unknown>;
    if (
      (wire.type === 'function_call' || wire.type === 'custom_tool_call') &&
      typeof wire.call_id === 'string' &&
      retiredCallIds.has(wire.call_id) &&
      'id' in wire
    ) {
      changed = true;
      const { id: _id, ...rest } = wire;
      return rest;
    }
    return item;
  });
  return changed ? { ...(payload as Record<string, unknown>), input } : payload;
}

const INTERLEAVED_THINKING_BETA = 'interleaved-thinking-2025-05-14';
export const GEMINI_SKIP_THOUGHT_SIGNATURE = 'skip_thought_signature_validator';

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Provider-aware wire payload sanitizer after history mutations:
 * - OpenAI / Codex / Azure Responses: strips `id` on retired `function_call` / `custom_tool_call` items.
 * - Google Gemini / Vertex (`contents`): injects `skip_thought_signature_validator` on the first
 *   unsigned `functionCall` part of each `model` step in the current user turn.
 * - Anthropic & Bedrock budget thinking (`thinking.type === 'enabled'`): omits `thinking` (and
 *   `interleaved-thinking-2025-05-14`) on an active tool-result continuation if an assistant step
 *   in the current user turn had its leading thinking block retired.
 */
export function sanitizeProviderPayload(
  payload: unknown,
  options: { retiredCallIds: ReadonlySet<string>; hasMutatedHistory?: boolean }
): unknown {
  const next = stripRetiredResponsesIds(payload, options.retiredCallIds);
  if (!options.hasMutatedHistory || !isObjectRecord(next)) return next;

  // Google Gemini / Vertex (`payload.contents`)
  if (Array.isArray(next.contents)) {
    const contents = next.contents as unknown[];
    let lastPlainUser = -1;
    for (let i = contents.length - 1; i >= 0; i--) {
      const item = contents[i];
      if (!isObjectRecord(item) || item.role !== 'user' || !Array.isArray(item.parts)) continue;
      const hasNonResponse = item.parts.some(
        (part) => isObjectRecord(part) && !('functionResponse' in part)
      );
      if (hasNonResponse) {
        lastPlainUser = i;
        break;
      }
    }
    let googleChanged = false;
    const updatedContents = contents.map((item, idx) => {
      if (idx <= lastPlainUser || !isObjectRecord(item) || item.role !== 'model') return item;
      if (!Array.isArray(item.parts)) return item;
      const firstFcIdx = item.parts.findIndex(
        (part) => isObjectRecord(part) && 'functionCall' in part
      );
      if (firstFcIdx < 0) return item;
      const firstFc = item.parts[firstFcIdx] as Record<string, unknown>;
      if (typeof firstFc.thoughtSignature === 'string' && firstFc.thoughtSignature.length > 0) {
        return item;
      }
      googleChanged = true;
      const parts = [...item.parts];
      parts[firstFcIdx] = { ...firstFc, thoughtSignature: GEMINI_SKIP_THOUGHT_SIGNATURE };
      return { ...item, parts };
    });
    return googleChanged ? { ...next, contents: updatedContents } : next;
  }

  // Anthropic Messages (`payload.messages` + `payload.thinking.type === 'enabled'`)
  // Active tool continuation: at least one `user` message after the last `assistant` message
  // carries a `tool_result` block (even when followed or coalesced with a steering text block).
  if (
    Array.isArray(next.messages) &&
    isObjectRecord(next.thinking) &&
    next.thinking.type === 'enabled'
  ) {
    const messages = next.messages as unknown[];
    const lastAssistantIndex = messages.findLastIndex(
      (msg) => isObjectRecord(msg) && msg.role === 'assistant'
    );
    const isToolContinuation =
      lastAssistantIndex >= 0 &&
      messages
        .slice(lastAssistantIndex + 1)
        .some(
          (msg) =>
            isObjectRecord(msg) &&
            msg.role === 'user' &&
            Array.isArray(msg.content) &&
            msg.content.some((block) => isObjectRecord(block) && block.type === 'tool_result')
        );
    if (isToolContinuation) {
      let lastPlainUser = -1;
      for (let i = lastAssistantIndex - 1; i >= 0; i--) {
        const msg = messages[i];
        if (!isObjectRecord(msg) || msg.role !== 'user') continue;
        const hasToolResult =
          Array.isArray(msg.content) &&
          msg.content.some((block) => isObjectRecord(block) && block.type === 'tool_result');
        if (!hasToolResult) {
          lastPlainUser = i;
          break;
        }
      }
      const missingLeadingThinking = messages.slice(lastPlainUser + 1).some((msg) => {
        if (!isObjectRecord(msg) || msg.role !== 'assistant') return false;
        if (!Array.isArray(msg.content) || msg.content.length === 0) return true;
        const first = msg.content[0];
        return (
          !isObjectRecord(first) ||
          (first.type !== 'thinking' && first.type !== 'redacted_thinking')
        );
      });
      if (missingLeadingThinking) {
        const { thinking: _thinking, betas, ...rest } = next;
        if (Array.isArray(betas)) {
          const filtered = betas.filter((beta) => beta !== INTERLEAVED_THINKING_BETA);
          return filtered.length > 0 ? { ...rest, betas: filtered } : rest;
        }
        return rest;
      }
    }
  }

  // AWS Bedrock Converse (`payload.messages` + `additionalModelRequestFields.thinking.type === 'enabled'`)
  // Bedrock appends `{ cachePoint: { type: 'default' } }` onto user messages when prompt caching
  // is active, so ignore `cachePoint` blocks when distinguishing tool results from user prompts.
  if (
    Array.isArray(next.messages) &&
    isObjectRecord(next.additionalModelRequestFields) &&
    isObjectRecord(next.additionalModelRequestFields.thinking) &&
    next.additionalModelRequestFields.thinking.type === 'enabled'
  ) {
    const messages = next.messages as unknown[];
    const lastAssistantIndex = messages.findLastIndex(
      (msg) => isObjectRecord(msg) && msg.role === 'assistant'
    );
    const isToolContinuation =
      lastAssistantIndex >= 0 &&
      messages
        .slice(lastAssistantIndex + 1)
        .some(
          (msg) =>
            isObjectRecord(msg) &&
            msg.role === 'user' &&
            Array.isArray(msg.content) &&
            msg.content.some((block) => isObjectRecord(block) && 'toolResult' in block)
        );
    if (isToolContinuation) {
      let lastPlainUser = -1;
      for (let i = lastAssistantIndex - 1; i >= 0; i--) {
        const msg = messages[i];
        if (!isObjectRecord(msg) || msg.role !== 'user' || !Array.isArray(msg.content)) continue;
        const nonCacheBlocks = msg.content.filter(
          (block) => !isObjectRecord(block) || !('cachePoint' in block)
        );
        if (
          nonCacheBlocks.length > 0 &&
          !nonCacheBlocks.some((block) => isObjectRecord(block) && 'toolResult' in block)
        ) {
          lastPlainUser = i;
          break;
        }
      }
      const missingLeadingReasoning = messages.slice(lastPlainUser + 1).some((msg) => {
        if (!isObjectRecord(msg) || msg.role !== 'assistant') return false;
        if (!Array.isArray(msg.content) || msg.content.length === 0) return true;
        const first = msg.content[0];
        return !isObjectRecord(first) || !('reasoningContent' in first);
      });
      if (missingLeadingReasoning) {
        const {
          thinking: _thinking,
          anthropic_beta,
          ...otherFields
        } = next.additionalModelRequestFields;
        if (Array.isArray(anthropic_beta)) {
          const filtered = anthropic_beta.filter((beta) => beta !== INTERLEAVED_THINKING_BETA);
          if (filtered.length > 0) otherFields.anthropic_beta = filtered;
        }
        const { additionalModelRequestFields: _old, ...rest } = next;
        return Object.keys(otherFields).length > 0
          ? { ...rest, additionalModelRequestFields: otherFields }
          : rest;
      }
    }
  }

  // OpenAI Chat Completions (`payload.messages` with `role: 'tool'` continuations — OpenRouter &
  // Google OpenAI-compatible endpoints)
  if (Array.isArray(next.messages) && typeof next.model === 'string') {
    const messages = next.messages as unknown[];
    const lastAssistantIndex = messages.findLastIndex(
      (msg) => isObjectRecord(msg) && msg.role === 'assistant'
    );
    const isToolContinuation =
      lastAssistantIndex >= 0 &&
      messages
        .slice(lastAssistantIndex + 1)
        .some((msg) => isObjectRecord(msg) && msg.role === 'tool');
    if (isToolContinuation) {
      let lastPlainUser = -1;
      for (let i = lastAssistantIndex - 1; i >= 0; i--) {
        const msg = messages[i];
        const prev = i > 0 ? messages[i - 1] : undefined;
        if (
          isObjectRecord(msg) &&
          msg.role === 'user' &&
          (!isObjectRecord(prev) || prev.role !== 'tool')
        ) {
          lastPlainUser = i;
          break;
        }
      }
      const openRouterReasoning = isObjectRecord(next.reasoning) ? next.reasoning : undefined;
      if (/gemini/i.test(next.model)) {
        let chatGeminiChanged = false;
        const updatedMessages = messages.map((msg, idx) => {
          if (idx <= lastPlainUser || !isObjectRecord(msg) || msg.role !== 'assistant') return msg;
          if (!Array.isArray(msg.tool_calls) || msg.tool_calls.length === 0) return msg;
          if (Array.isArray(msg.reasoning_details) && msg.reasoning_details.length > 0) return msg;
          const firstTc = msg.tool_calls[0];
          if (!isObjectRecord(firstTc)) return msg;
          if (
            isObjectRecord(firstTc.extra_content) &&
            isObjectRecord(firstTc.extra_content.google) &&
            typeof firstTc.extra_content.google.thought_signature === 'string' &&
            firstTc.extra_content.google.thought_signature.length > 0
          ) {
            return msg;
          }
          chatGeminiChanged = true;
          if (openRouterReasoning !== undefined) {
            return {
              ...msg,
              reasoning_details: [
                {
                  type: 'reasoning.encrypted',
                  data: GEMINI_SKIP_THOUGHT_SIGNATURE,
                  ...(typeof firstTc.id === 'string' ? { id: firstTc.id } : {}),
                  format: 'google-gemini-v1',
                  index: 0
                }
              ]
            };
          }
          const toolCalls = [...msg.tool_calls];
          toolCalls[0] = {
            ...firstTc,
            extra_content: {
              ...(isObjectRecord(firstTc.extra_content) ? firstTc.extra_content : {}),
              google: {
                ...(isObjectRecord(firstTc.extra_content) &&
                isObjectRecord(firstTc.extra_content.google)
                  ? firstTc.extra_content.google
                  : {}),
                thought_signature: GEMINI_SKIP_THOUGHT_SIGNATURE
              }
            }
          };
          return { ...msg, tool_calls: toolCalls };
        });
        if (chatGeminiChanged) return { ...next, messages: updatedMessages };
      }
      if (
        /claude/i.test(next.model) &&
        openRouterReasoning !== undefined &&
        openRouterReasoning.effort !== 'none'
      ) {
        const missingReasoningDetails = messages.slice(lastPlainUser + 1).some((msg) => {
          if (!isObjectRecord(msg) || msg.role !== 'assistant') return false;
          return !Array.isArray(msg.reasoning_details) || msg.reasoning_details.length === 0;
        });
        if (missingReasoningDetails) {
          const { reasoning: _reasoning, ...rest } = next;
          return rest;
        }
      }
    }
  }

  return next;
}
