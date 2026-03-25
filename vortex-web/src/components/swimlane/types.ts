// Layout node types for the columnar format visualizer

export type DtypeCategory = 'bool' | 'int' | 'float' | 'struct' | 'list' | 'other';

export type LayoutType = 'struct' | 'chunked' | 'zonemap' | 'dict' | 'flat';

export interface FlatMeta {
  dtype: string;
  bytes: number;
  min?: string | number | boolean;
  max?: string | number | boolean;
}

export interface ZoneMeta {
  min: number;
  max: number;
}

export interface BaseLayoutNode {
  id: string;
  name: string;
  rowRange: [number, number];
}

export interface StructLayout extends BaseLayoutNode {
  type: 'struct';
  children: LayoutNode[];
}

export interface ChunkedLayout extends BaseLayoutNode {
  type: 'chunked';
  chunkCount: number;
  chunks: ChunkNode[];
}

export interface ZonemapLayout extends BaseLayoutNode {
  type: 'zonemap';
  zoneCount: number;
  zones: ZoneNode[];
}

export interface DictLayout extends BaseLayoutNode {
  type: 'dict';
  children: LayoutNode[];
}

export interface FlatLayout extends BaseLayoutNode {
  type: 'flat';
  meta: FlatMeta;
}

export interface ChunkNode extends BaseLayoutNode {
  child: LayoutNode;
}

export interface ZoneNode extends BaseLayoutNode {
  meta: ZoneMeta;
  child: LayoutNode;
}

export type LayoutNode = 
  | StructLayout 
  | ChunkedLayout 
  | ZonemapLayout 
  | DictLayout 
  | FlatLayout;

export interface Split {
  id: string;
  rowRange: [number, number];
}

// Internal types for tree flattening
export interface FlattenedNode {
  node: LayoutNode & { 
    _isPartition?: boolean; 
    _isGroup?: boolean;
    chunks?: ChunkNode[];
  };
  depth: number;
  isGroup?: boolean;
  isHint?: boolean;
  isHiddenIndicator?: boolean;
}

export interface ChunkGroup {
  id: string;
  name: string;
  type: 'chunkGroup';
  rowRange: [number, number];
  chunks: ChunkNode[];
  _isGroup: true;
}
