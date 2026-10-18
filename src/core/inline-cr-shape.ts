/**
 * Contextual-retrieval shape shared by the inline import and the deferred
 * embed passes (`gbrain embed --stale`, the cycle's embed phase, the
 * `embed-backfill` Minion, `gbrain embed <slug>` and `--all`).
 *
 * Why this module exists (Atlas fork, 2026-10-18, cf-ops-258): import-file.ts
 * resolved the CR mode, wrapped chunks with the title prefix and stamped
 * `corpus_generation` itself, and its noEmbed branch promised that "the next
 * embed pass (gbrain embed --stale or contextual reindex Minion) will set it".
 * No embed pass did. They embedded raw chunk_text and never stamped, so every
 * page the autopilot cycle imported (sync with noEmbed, then a separate embed
 * phase) landed with NULL corpus_generation and vectors without the title
 * wrapper the rest of the corpus has. One resolver for every path keeps the
 * inline and deferred results identical.
 *
 * The per-chunk synopsis tier is never applied here: like the inline import,
 * these paths take the free title tier and leave Haiku synopses to the
 * contextual reindex Minion.
 *
 * Known limit: a partial repair of a page the reindex Minion already upgraded
 * to per_chunk_synopsis embeds the stale chunks title-only and keeps the
 * page's synopsis stamp (before this module those chunks were embedded as raw
 * text, which was also mismatched). Fixing it needs an engine read of the
 * page's stored mode. No synopsis-tier pages existed on the Atlas brain when
 * this landed (2026-10-18: 2565 title, 3075 unstamped).
 */

import type { BrainEngine } from './engine.ts';
import type { CRMode } from './types.ts';
import { resolveContextualRetrievalMode } from './contextual-retrieval-resolver.ts';
import { loadSearchModeConfig, resolveSearchMode } from './search/mode.ts';
import { computeCorpusGeneration } from './contextual-retrieval-service.ts';
import {
  buildContextualPrefix,
  modeRequiresHaiku,
  modeRequiresWrapper,
  sanitizeTitle,
  wrapChunkForEmbedding,
} from './embedding-context.ts';

/** Haiku model folded into corpus_generation on the inline/title paths. */
export const INLINE_HAIKU_MODEL = 'anthropic:claude-haiku-4-5-20251001';

export interface InlineCRKnobs {
  globalMode: CRMode;
  killSwitchDisabled: boolean;
}

export interface InlineCRShape {
  /** The tier actually applied (per_chunk_synopsis is demoted to title). */
  mode: CRMode;
  /** Title-only prefix for the embedding input, or null for no wrapper. */
  prefix: string | null;
  /** Value for pages.corpus_generation; null when no wrapper applies. */
  corpusGeneration: string | null;
}

/** Load the global CR knobs once per run; per-page resolution is pure. */
export async function loadInlineCRKnobs(
  engine: Pick<BrainEngine, 'getConfig'>,
): Promise<InlineCRKnobs> {
  const knobs = resolveSearchMode(await loadSearchModeConfig(engine));
  return {
    globalMode: knobs.contextual_retrieval,
    killSwitchDisabled: knobs.contextual_retrieval_disabled,
  };
}

export function resolveInlineCRShape(args: {
  frontmatter: Record<string, unknown> | null | undefined;
  title: string;
  sourceId: string | undefined;
  knobs: InlineCRKnobs;
}): InlineCRShape {
  // Default to host trust: the inline import never surfaced a source row,
  // and the deferred paths must land on the same result it would have.
  const resolution = resolveContextualRetrievalMode({
    pageFrontmatter: args.frontmatter ?? {},
    source: {
      id: args.sourceId ?? 'default',
      contextual_retrieval_mode: null,
      trust_frontmatter_overrides: false,
    },
    globalMode: args.knobs.globalMode,
    killSwitchDisabled: args.knobs.killSwitchDisabled,
  });
  const mode: CRMode = resolution.mode === 'per_chunk_synopsis' ? 'title' : resolution.mode;
  const prefix =
    modeRequiresWrapper(mode) && !modeRequiresHaiku(mode)
      ? buildContextualPrefix(sanitizeTitle(args.title), null)
      : null;
  const corpusGeneration =
    mode === 'none' ? null : computeCorpusGeneration({ crMode: mode, haikuModel: INLINE_HAIKU_MODEL });
  return { mode, prefix, corpusGeneration };
}

/**
 * Shape for one stored page, for the embed passes. Returns null when the
 * page row is missing (or the engine can't load it), so the caller falls
 * back to raw text and leaves the stamp alone.
 */
export async function pageInlineCRShape(
  engine: BrainEngine,
  slug: string,
  sourceId: string | undefined,
  knobs: InlineCRKnobs | null,
): Promise<InlineCRShape | null> {
  if (!knobs) return null;
  const page = await engine.getPage(slug, sourceId ? { sourceId } : undefined);
  if (!page) return null;
  return resolveInlineCRShape({
    frontmatter: page.frontmatter,
    title: page.title,
    sourceId,
    knobs,
  });
}

/** The strings to send to the embedder; stored chunk_text stays unwrapped. */
export function embeddingInputsFor(
  chunks: ReadonlyArray<{ chunk_text: string; chunk_source?: string | null }>,
  shape: InlineCRShape | null,
): string[] {
  const prefix = shape?.prefix ?? null;
  return chunks.map((c) => wrapChunkForEmbedding(c.chunk_text, prefix, c.chunk_source));
}

/** Record the shape a fully re-embedded page now carries. */
export async function stampInlineCRShape(
  engine: BrainEngine,
  slug: string,
  sourceId: string | undefined,
  shape: InlineCRShape,
): Promise<void> {
  await engine.updatePageContextualRetrievalState(
    slug,
    sourceId ?? 'default',
    shape.mode,
    shape.corpusGeneration,
  );
}

/** Per-run coverage state: chunk indexes embedded per page, under one shape. */
export type PassCoverage = Map<string, { shapeId: string; covered: Set<number> }>;

/** Identity of the wrapper a page was embedded under (raw text when null). */
export function inlineCRShapeId(shape: InlineCRShape | null): string {
  return shape ? `${shape.mode}|${shape.corpusGeneration ?? ''}|${shape.prefix ?? ''}` : 'raw';
}

/**
 * Whole-pass coverage check for the stamps (CR shape and embedding
 * signature). Keyset batches can split one page's stale chunks, so a stamp
 * is earned only when this pass embedded every chunk the page has now, all
 * under the same shape, AND none is left unembedded: a page re-imported or
 * retitled mid-pass is not certified. A failed or aborted pass leaves the
 * page unstamped, which is the safe side: the freshness watchdog reports it
 * and `gbrain reindex` repairs it. A certified page's entry is dropped.
 */
export function pageFullyEmbeddedThisPass(
  coverage: PassCoverage,
  key: string,
  shapeId: string,
  justEmbedded: ReadonlyArray<number>,
  current: ReadonlyArray<{ chunk_index: number; embedded_at?: unknown }>,
): boolean {
  let entry = coverage.get(key);
  if (!entry || entry.shapeId !== shapeId) {
    entry = { shapeId, covered: new Set<number>() };
    coverage.set(key, entry);
  }
  for (const i of justEmbedded) entry.covered.add(i);
  const now = new Set(justEmbedded);
  const covered = entry.covered;
  const full = current.every(
    (c) => covered.has(c.chunk_index) && (now.has(c.chunk_index) || c.embedded_at != null),
  );
  if (full) coverage.delete(key);
  return full;
}

/**
 * Bound the coverage map to the one page that can still continue: keyset
 * order is (page_id, chunk_index), so only the last page of a batch can have
 * more stale chunks in the next one.
 */
export function prunePassCoverage(coverage: PassCoverage, keepKey: string | undefined): void {
  for (const k of coverage.keys()) if (k !== keepKey) coverage.delete(k);
}
