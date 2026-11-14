import { createHash, randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { ImageContent, TextContent } from '@earendil-works/pi-ai';
import { isStampBlock } from './stamps.js';

export type SupportedImageMime = 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp';

const IMAGE_EXTENSIONS: Record<SupportedImageMime, string> = {
  'image/png': '.png',
  'image/jpeg': '.jpg',
  'image/gif': '.gif',
  'image/webp': '.webp'
};

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] as const;

export interface ArchivedImage {
  /** Handle of the fold whose manifest archived this image, e.g. "ep-1a2b3c4d". */
  handle: string;
  /** 1-based image index within `handle`'s own archived images. */
  index: number;
  /** Absolute path to the archived image file in `<session>.episodes/`. */
  path: string;
  /** Full 64-character hex SHA-256 digest of the raw decoded image bytes. */
  sha256: string;
  /** Verified MIME type matching both the block and the file header. */
  mimeType: SupportedImageMime;
  /** Raw byte length of the archived image. */
  bytes: number;
  /** Pixel width when reliably parsed from the image header. */
  width?: number;
  /** Pixel height when reliably parsed from the image header. */
  height?: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function normalizeImageMime(mimeType: string): SupportedImageMime | undefined {
  const base = mimeType.split(';')[0]?.trim().toLowerCase();
  switch (base) {
    case 'image/png':
      return 'image/png';
    case 'image/jpeg':
    case 'image/jpg':
      return 'image/jpeg';
    case 'image/gif':
      return 'image/gif';
    case 'image/webp':
      return 'image/webp';
    default:
      return undefined;
  }
}

export function isSupportedImageMime(value: unknown): value is SupportedImageMime {
  return (
    value === 'image/png' ||
    value === 'image/jpeg' ||
    value === 'image/gif' ||
    value === 'image/webp'
  );
}

function readUint16BE(bytes: Uint8Array, offset: number): number {
  return ((bytes[offset] ?? 0) << 8) | (bytes[offset + 1] ?? 0);
}

function readUint16LE(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] ?? 0) | ((bytes[offset + 1] ?? 0) << 8);
}

function readUint32BE(bytes: Uint8Array, offset: number): number {
  return (
    (bytes[offset] ?? 0) * 0x1000000 +
    ((bytes[offset + 1] ?? 0) << 16) +
    ((bytes[offset + 2] ?? 0) << 8) +
    (bytes[offset + 3] ?? 0)
  );
}

function readUint32LE(bytes: Uint8Array, offset: number): number {
  return (
    (bytes[offset] ?? 0) +
    ((bytes[offset + 1] ?? 0) << 8) +
    ((bytes[offset + 2] ?? 0) << 16) +
    (bytes[offset + 3] ?? 0) * 0x1000000
  );
}

function startsWithAscii(bytes: Uint8Array, offset: number, text: string): boolean {
  if (bytes.length < offset + text.length) return false;
  for (let i = 0; i < text.length; i++) {
    if (bytes[offset + i] !== text.charCodeAt(i)) return false;
  }
  return true;
}

function isAnimatedPng(bytes: Uint8Array): boolean {
  let offset: number = PNG_SIGNATURE.length;
  while (offset + 8 <= bytes.length) {
    const chunkLength = readUint32BE(bytes, offset);
    if (startsWithAscii(bytes, offset + 4, 'acTL')) return true;
    if (startsWithAscii(bytes, offset + 4, 'IDAT')) return false;
    const next = offset + 8 + chunkLength + 4;
    if (next <= offset || next > bytes.length) return false;
    offset = next;
  }
  return false;
}

export function sniffImageMimeType(bytes: Uint8Array): SupportedImageMime | undefined {
  if (
    bytes.length >= 4 &&
    bytes[0] === 0xff &&
    bytes[1] === 0xd8 &&
    bytes[2] === 0xff &&
    bytes[3] !== 0xf7
  ) {
    return 'image/jpeg';
  }
  if (
    bytes.length >= 24 &&
    PNG_SIGNATURE.every((b, idx) => bytes[idx] === b) &&
    readUint32BE(bytes, 8) === 13 &&
    startsWithAscii(bytes, 12, 'IHDR') &&
    !isAnimatedPng(bytes)
  ) {
    return 'image/png';
  }
  if (
    bytes.length >= 10 &&
    (startsWithAscii(bytes, 0, 'GIF87a') || startsWithAscii(bytes, 0, 'GIF89a'))
  ) {
    return 'image/gif';
  }
  if (
    bytes.length >= 16 &&
    startsWithAscii(bytes, 0, 'RIFF') &&
    startsWithAscii(bytes, 8, 'WEBP')
  ) {
    return 'image/webp';
  }
  return undefined;
}

function jpegExifOrientation(
  bytes: Uint8Array,
  tiffStart: number,
  limit: number
): number | undefined {
  if (tiffStart + 8 > limit) return undefined;
  const little =
    bytes[tiffStart] === 0x49 && bytes[tiffStart + 1] === 0x49
      ? true
      : bytes[tiffStart] === 0x4d && bytes[tiffStart + 1] === 0x4d
        ? false
        : undefined;
  if (little === undefined) return undefined;
  const u16 = (off: number) => (little ? readUint16LE(bytes, off) : readUint16BE(bytes, off));
  const u32 = (off: number) => (little ? readUint32LE(bytes, off) : readUint32BE(bytes, off));
  if (u16(tiffStart + 2) !== 0x002a) return undefined;
  const ifdOffset = u32(tiffStart + 4);
  const ifdStart = tiffStart + ifdOffset;
  if (ifdOffset < 8 || ifdStart + 2 > limit) return undefined;
  const entries = u16(ifdStart);
  for (let i = 0; i < entries; i++) {
    const entry = ifdStart + 2 + i * 12;
    if (entry + 12 > limit) break;
    if (u16(entry) === 0x0112 && u16(entry + 2) === 3) {
      return u16(entry + 8);
    }
  }
  return undefined;
}

function parseJpegDimensions(bytes: Uint8Array): { width: number; height: number } | undefined {
  let pos = 2;
  let orientation: number | undefined;
  while (pos < bytes.length) {
    if (bytes[pos] !== 0xff) {
      pos++;
      continue;
    }
    while (pos < bytes.length && bytes[pos] === 0xff) pos++;
    if (pos >= bytes.length) break;
    const marker = bytes[pos]!;
    pos++;
    if (marker === 0xd9 || marker === 0xda) break;
    if (marker === 0x00 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd8)) continue;
    if (pos + 2 > bytes.length) break;
    const segLen = readUint16BE(bytes, pos);
    if (segLen < 2 || pos + segLen > bytes.length) break;
    if (
      marker === 0xe1 &&
      orientation === undefined &&
      segLen >= 14 &&
      startsWithAscii(bytes, pos + 2, 'Exif') &&
      bytes[pos + 6] === 0 &&
      bytes[pos + 7] === 0
    ) {
      orientation = jpegExifOrientation(bytes, pos + 8, pos + segLen);
    }
    const isSof =
      (marker >= 0xc0 && marker <= 0xc3) ||
      (marker >= 0xc5 && marker <= 0xc7) ||
      (marker >= 0xc9 && marker <= 0xcb) ||
      (marker >= 0xcd && marker <= 0xcf);
    if (isSof) {
      if (segLen < 7) return undefined;
      const rawHeight = readUint16BE(bytes, pos + 3);
      const rawWidth = readUint16BE(bytes, pos + 5);
      if (rawWidth <= 0 || rawHeight <= 0) return undefined;
      const swapped = orientation !== undefined && orientation >= 5 && orientation <= 8;
      return swapped
        ? { width: rawHeight, height: rawWidth }
        : { width: rawWidth, height: rawHeight };
    }
    pos += segLen;
  }
  return undefined;
}

function parseWebpDimensions(bytes: Uint8Array): { width: number; height: number } | undefined {
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const chunkSize = readUint32LE(bytes, offset + 4);
    const dataStart = offset + 8;
    if (startsWithAscii(bytes, offset, 'VP8X') && dataStart + 10 <= bytes.length) {
      const width =
        ((bytes[dataStart + 4] ?? 0) |
          ((bytes[dataStart + 5] ?? 0) << 8) |
          ((bytes[dataStart + 6] ?? 0) << 16)) +
        1;
      const height =
        ((bytes[dataStart + 7] ?? 0) |
          ((bytes[dataStart + 8] ?? 0) << 8) |
          ((bytes[dataStart + 9] ?? 0) << 16)) +
        1;
      return width > 0 && height > 0 ? { width, height } : undefined;
    }
    if (startsWithAscii(bytes, offset, 'VP8L') && dataStart + 5 <= bytes.length) {
      if (bytes[dataStart] !== 0x2f) return undefined;
      const bits = readUint32LE(bytes, dataStart + 1);
      const width = (bits & 0x3fff) + 1;
      const height = ((bits >>> 14) & 0x3fff) + 1;
      return width > 0 && height > 0 ? { width, height } : undefined;
    }
    if (startsWithAscii(bytes, offset, 'VP8 ') && dataStart + 10 <= bytes.length) {
      if (
        (bytes[dataStart] ?? 1) & 1 ||
        bytes[dataStart + 3] !== 0x9d ||
        bytes[dataStart + 4] !== 0x01 ||
        bytes[dataStart + 5] !== 0x2a
      ) {
        return undefined;
      }
      const width = readUint16LE(bytes, dataStart + 6) & 0x3fff;
      const height = readUint16LE(bytes, dataStart + 8) & 0x3fff;
      return width > 0 && height > 0 ? { width, height } : undefined;
    }
    const padded = chunkSize + (chunkSize & 1);
    const next = dataStart + padded;
    if (next <= offset) break;
    offset = next;
  }
  return undefined;
}

export function imageDimensions(
  bytes: Uint8Array,
  mimeType: SupportedImageMime
): { width: number; height: number } | undefined {
  switch (mimeType) {
    case 'image/png': {
      if (bytes.length < 24) return undefined;
      const width = readUint32BE(bytes, 16);
      const height = readUint32BE(bytes, 20);
      return width > 0 && height > 0 ? { width, height } : undefined;
    }
    case 'image/gif': {
      if (bytes.length < 10) return undefined;
      const width = readUint16LE(bytes, 6);
      const height = readUint16LE(bytes, 8);
      return width > 0 && height > 0 ? { width, height } : undefined;
    }
    case 'image/jpeg':
      return parseJpegDimensions(bytes);
    case 'image/webp':
      return parseWebpDimensions(bytes);
  }
}

/**
 * Strictly decodes canonical base64 image data and verifies that its binary header matches its
 * declared MIME type. Returns `undefined` for malformed base64, unsupported MIME types, or
 * mismatched headers so callers leave unrecognized payloads inline without loss.
 */
export function decodeImageBlock(
  image: ImageContent
): { bytes: Buffer; mimeType: SupportedImageMime } | undefined {
  if (
    image.type !== 'image' ||
    typeof image.data !== 'string' ||
    typeof image.mimeType !== 'string'
  ) {
    return undefined;
  }
  const declared = normalizeImageMime(image.mimeType);
  if (!declared) return undefined;
  const clean = image.data.replace(/\s+/g, '');
  if (clean.length === 0 || clean.length % 4 !== 0 || !/^[A-Za-z0-9+/]+={0,2}$/.test(clean)) {
    return undefined;
  }
  const bytes = Buffer.from(clean, 'base64');
  if (bytes.length === 0 || bytes.toString('base64') !== clean) return undefined;
  const sniffed = sniffImageMimeType(bytes);
  if (!sniffed || sniffed !== declared) return undefined;
  return { bytes, mimeType: sniffed };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatImageRef(ref: ArchivedImage): string {
  const dims =
    ref.width !== undefined && ref.height !== undefined ? `, ${ref.width}x${ref.height}` : '';
  return `[${ref.handle} image #${ref.index}: ${ref.mimeType}${dims}, ${formatBytes(ref.bytes)}, sha256 ${ref.sha256.slice(0, 16)} — ${ref.path}; recall({ handle: "${ref.handle}", image: ${ref.index} }) or read path]`;
}

export function isArchivedImage(value: unknown): value is ArchivedImage {
  return (
    isRecord(value) &&
    typeof value.handle === 'string' &&
    value.handle.length > 0 &&
    typeof value.index === 'number' &&
    Number.isInteger(value.index) &&
    value.index >= 1 &&
    typeof value.path === 'string' &&
    value.path.length > 0 &&
    typeof value.sha256 === 'string' &&
    /^[0-9a-f]{64}$/.test(value.sha256) &&
    isSupportedImageMime(value.mimeType) &&
    typeof value.bytes === 'number' &&
    Number.isInteger(value.bytes) &&
    value.bytes > 0 &&
    (value.width === undefined || (typeof value.width === 'number' && value.width > 0)) &&
    (value.height === undefined || (typeof value.height === 'number' && value.height > 0))
  );
}

export function archivedImagesOf(value: unknown): ArchivedImage[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isArchivedImage);
}

/**
 * Atomically writes an image's original bytes under `dir` using a content-addressed filename
 * before returning its manifest entry. Returns `undefined` when `dir` is absent, the image has
 * unsupported or malformed bytes, or writing to disk fails.
 */
export async function archiveImage(
  dir: string | undefined,
  handle: string,
  index: number,
  image: ImageContent
): Promise<ArchivedImage | undefined> {
  if (!dir) return undefined;
  const decoded = decodeImageBlock(image);
  if (!decoded) return undefined;
  const { bytes, mimeType } = decoded;
  const digest = createHash('sha256').update(bytes).digest('hex');
  const ext = IMAGE_EXTENSIONS[mimeType];
  const path = join(dir, `image-${digest.slice(0, 16)}${ext}`);
  const temporary = `${path}.${randomUUID()}.tmp`;
  try {
    await mkdir(dir, { recursive: true });
    await writeFile(temporary, bytes);
    await rename(temporary, path);
  } catch {
    await rm(temporary, { force: true }).catch(() => undefined);
    return undefined;
  }
  const dims = imageDimensions(bytes, mimeType);
  return {
    handle,
    index,
    path,
    sha256: digest,
    mimeType,
    bytes: bytes.length,
    ...(dims ? { width: dims.width, height: dims.height } : {})
  };
}

/** Synchronous integrity check used at `turn_end` before committing fold edits. */
export function verifyArchivedImageSync(ref: ArchivedImage): boolean {
  try {
    const bytes = readFileSync(ref.path);
    if (bytes.length !== ref.bytes) return false;
    if (sniffImageMimeType(bytes) !== ref.mimeType) return false;
    return createHash('sha256').update(bytes).digest('hex') === ref.sha256;
  } catch {
    return false;
  }
}

/** Reads and verifies an archived image from disk, throwing an explicit error if missing or corrupt. */
export async function readArchivedImage(ref: ArchivedImage): Promise<ImageContent> {
  let bytes: Buffer;
  try {
    bytes = await readFile(ref.path);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException | undefined)?.code;
    if (code === 'ENOENT') {
      throw new Error(`Archived image #${ref.index} of ${ref.handle} is missing at ${ref.path}.`);
    }
    throw new Error(
      `Archived image #${ref.index} of ${ref.handle} could not be read at ${ref.path}: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  const actualSha = createHash('sha256').update(bytes).digest('hex');
  if (
    bytes.length !== ref.bytes ||
    actualSha !== ref.sha256 ||
    sniffImageMimeType(bytes) !== ref.mimeType
  ) {
    throw new Error(
      `Archived image #${ref.index} of ${ref.handle} at ${ref.path} failed SHA-256 verification (expected ${ref.sha256.slice(0, 16)}).`
    );
  }
  return {
    type: 'image',
    data: bytes.toString('base64'),
    mimeType: ref.mimeType
  };
}

function withoutTrailingStamp(
  content: readonly (TextContent | ImageContent)[]
): readonly (TextContent | ImageContent)[] {
  const last = content.at(-1);
  return last && isStampBlock(last) ? content.slice(0, -1) : content;
}

/**
 * Matches standalone text blocks in `blocks` (1-to-1, in block order) whose `text` equals
 * `formatImageRef(ref)` for an entry in `archived`. Modified blocks, substring embeds, and blocks
 * without a manifest entry are left unmatched so callers preserve them as ordinary text.
 */
export function matchArchivedRefBlocks(
  blocks: readonly (TextContent | ImageContent)[],
  archived: readonly ArchivedImage[]
): Map<number, ArchivedImage> {
  const matched = new Map<number, ArchivedImage>();
  if (archived.length === 0) return matched;
  const remaining = [...archived];
  for (let i = 0; i < blocks.length && remaining.length > 0; i++) {
    const block = blocks[i];
    if (block?.type !== 'text') continue;
    const hit = remaining.findIndex((ref) => block.text === formatImageRef(ref));
    if (hit >= 0) {
      matched.set(i, remaining[hit]!);
      remaining.splice(hit, 1);
    }
  }
  return matched;
}

/** Whether `content` holds `ref` as an exact standalone text block. */
export function holdsImageReference(
  content: readonly (TextContent | ImageContent)[],
  ref: ArchivedImage
): boolean {
  const expected = formatImageRef(ref);
  return withoutTrailingStamp(content).some(
    (block) => block.type === 'text' && block.text === expected
  );
}

export type CarriedMediaInput =
  | { kind: 'image'; image: ImageContent }
  | { kind: 'ref'; ref: ArchivedImage };

export type PreparedMediaItem =
  | { kind: 'image'; image: ImageContent }
  | { kind: 'ref'; ref: ArchivedImage; original?: ImageContent };

/**
 * Prepares covered-note media items for `archivingHandle`: existing archive references retain
 * their original `(handle, index)` identity unchanged, and newly offloaded inline images receive
 * immutable sequential `(archivingHandle, index)` identities (or stay inline on fallback).
 */
export async function prepareCarriedMedia(
  archivingHandle: string,
  items: readonly CarriedMediaInput[],
  dir: string | undefined
): Promise<PreparedMediaItem[]> {
  const prepared: PreparedMediaItem[] = [];
  let nextIndex = 1;
  for (const item of items) {
    if (item.kind === 'ref' && item.ref.handle === archivingHandle && item.ref.index >= nextIndex) {
      nextIndex = item.ref.index + 1;
    }
  }
  for (const item of items) {
    if (item.kind === 'ref') {
      prepared.push({ kind: 'ref', ref: item.ref });
      continue;
    }
    const archived = await archiveImage(dir, archivingHandle, nextIndex, item.image);
    if (archived) {
      nextIndex++;
      prepared.push({ kind: 'ref', ref: archived, original: item.image });
    } else {
      prepared.push({ kind: 'image', image: item.image });
    }
  }
  return prepared;
}

export function carriedMediaBlocks(
  items: readonly PreparedMediaItem[]
): (TextContent | ImageContent)[] {
  return items.map((item) =>
    item.kind === 'image' ? item.image : { type: 'text', text: formatImageRef(item.ref) }
  );
}

/**
 * Verifies at `turn_end` (1-to-1 across standalone blocks) that every carried media item survived
 * on the fold result either as its verified standalone archive reference block or as its original
 * inline image block.
 */
export function holdsCarriedMedia(
  content: readonly (TextContent | ImageContent)[],
  items: readonly PreparedMediaItem[]
): boolean {
  const blocks = withoutTrailingStamp(content);
  const used = new Set<number>();
  const takeImage = (target: ImageContent): boolean => {
    for (let i = 0; i < blocks.length; i++) {
      if (used.has(i)) continue;
      const block = blocks[i];
      if (
        block?.type === 'image' &&
        block.data === target.data &&
        block.mimeType === target.mimeType
      ) {
        used.add(i);
        return true;
      }
    }
    return false;
  };
  const takeRef = (ref: ArchivedImage): boolean => {
    const expected = formatImageRef(ref);
    for (let i = 0; i < blocks.length; i++) {
      if (used.has(i)) continue;
      const block = blocks[i];
      if (block?.type === 'text' && block.text === expected) {
        used.add(i);
        return true;
      }
    }
    return false;
  };
  return items.every((item) => {
    if (item.kind === 'image') return takeImage(item.image);
    if (item.original !== undefined) {
      if (verifyArchivedImageSync(item.ref) && takeRef(item.ref)) return true;
      return takeImage(item.original);
    }
    return takeRef(item.ref);
  });
}
