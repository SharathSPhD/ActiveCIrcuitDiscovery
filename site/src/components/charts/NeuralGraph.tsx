import { h, Fragment } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import * as d3 from 'd3';
import graph from '../../data/graphs/ioi_gemma.json';

interface NodeData {
  id: string;
  layer: number | null;
  feature: number;
  ctx: number;
  influence: number;
  activation: number;
  logit: boolean;
}

interface LinkData {
  s: string;
  t: string;
  w: number;
}

interface SimNode extends NodeData {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface SimLink {
  source: SimNode;
  target: SimNode;
  w: number;
}

interface NeuralGraphProps {
  height?: number;
}

/**
 * NeuralGraph: Interactive d3-force directed graph of transformer attribution subgraph.
 * Renders nodes colored by layer depth, scaled by influence. Links show weights with opacity.
 * Hover/focus nodes to highlight connected neighbors. Respects prefers-reduced-motion.
 * SSR-safe: all d3/DOM work in useEffect, renders empty container on server.
 */
export default function NeuralGraph({ height = 520 }: NeuralGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const prefersReducedMotion = typeof window !== 'undefined'
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
    : false;

  // Measure container width on mount and resize
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const updateWidth = () => {
      if (containerRef.current) {
        const w = containerRef.current.clientWidth || 800;
        setWidth(w);
      }
    };

    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Main d3 simulation and rendering
  useEffect(() => {
    if (typeof window === 'undefined' || !svgRef.current || width < 100) return;

    // Deep copy nodes to avoid mutation by d3-force
    const nodesCopy: SimNode[] = graph.nodes.map((node) => ({
      ...node,
      x: undefined,
      y: undefined,
    }));

    // Build id->node map for link remapping
    const nodeMap = new Map<string, SimNode>();
    nodesCopy.forEach((node) => {
      nodeMap.set(node.id, node);
    });

    // Deep copy links and remap s/t to node references
    const linksCopy: SimLink[] = graph.links
      .map(({ s, t, w }) => ({
        source: nodeMap.get(s)!,
        target: nodeMap.get(t)!,
        w,
      }))
      .filter(
        (link) => link.source && link.target // Skip broken links
      );

    // Color scale: early layers cyan, late/logit violet
    const maxLayer = Math.max(...nodesCopy.map((n) => n.layer ?? 0));
    const colorScale = d3
      .scaleLinear<string>()
      .domain([0, maxLayer])
      .range(['#22d3ee', '#a855f7']);

    // Influence to radius scale: [0,1] -> [3,10]px
    const radiusScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([3, 10])
      .clamp(true);

    // Weight to opacity scale for links
    const opacityScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([0.1, 0.6])
      .clamp(true);

    // Y position by layer (info flows upward: early at top, late+logit at bottom)
    const layerPositions = new Map<number | null, number>();
    const uniqueLayers = Array.from(new Set(nodesCopy.map((n) => n.layer))).sort(
      (a, b) => (a ?? Infinity) - (b ?? Infinity)
    );
    uniqueLayers.forEach((layer, i) => {
      layerPositions.set(layer, (i / (uniqueLayers.length - 1)) * height);
    });

    // Build d3-force simulation
    const simulation = d3
      .forceSimulation<SimNode>(nodesCopy)
      .force(
        'link',
        d3
          .forceLink<SimNode, SimLink>(linksCopy)
          .id((d) => d.id)
          .distance(40)
          .strength(0.1)
      )
      .force('charge', d3.forceManyBody().strength(-50))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force(
        'y',
        d3
          .forceY<SimNode>()
          .y((d) => layerPositions.get(d.layer) ?? height / 2)
          .strength(0.2)
      )
      .force('x', d3.forceX(width / 2).strength(0.05));

    const svg = d3.select(svgRef.current);

    // Clear previous content
    svg.selectAll('*').remove();

    // Define glow filter (shared for all nodes/links)
    const defs = svg.append('defs');
    const filter = defs
      .append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');
    filter
      .append('feGaussianBlur')
      .attr('stdDeviation', 2)
      .attr('result', 'coloredBlur');
    filter
      .append('feMerge')
      .selectAll('feMergeNode')
      .data([0, 1])
      .join('feMergeNode')
      .attr('in', (d) => (d ? 'SourceGraphic' : 'coloredBlur'));

    // Render links
    const linkSelection = svg
      .selectAll('.link')
      .data(linksCopy)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', (d) => (d.w > 0 ? '#22d3ee' : '#a855f7'))
      .attr('stroke-opacity', (d) => opacityScale(Math.abs(d.w)))
      .attr('stroke-width', 1)
      .attr('filter', 'url(#glow)');

    // Render nodes
    const nodeSelection = svg
      .selectAll('.node')
      .data(nodesCopy)
      .join('g')
      .attr('class', 'node')
      .attr('tabindex', 0) // Keyboard accessible
      .style('cursor', 'pointer')
      .style('outline', 'none');

    // Node circles
    nodeSelection
      .append('circle')
      .attr('r', (d) => radiusScale(d.influence))
      .attr('fill', (d) => (d.logit ? '#f59e0b' : colorScale(d.layer ?? 0)))
      .attr('filter', 'url(#glow)')
      .attr('data-testid', 'node');

    // Logit node rings (larger, outer stroke)
    nodeSelection
      .filter((d) => d.logit)
      .append('circle')
      .attr('r', (d) => radiusScale(d.influence) + 4)
      .attr('fill', 'none')
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.6);

    // Tooltip container (initially hidden)
    const tooltipDiv = d3
      .select(document.body)
      .append('div')
      .style('position', 'absolute')
      .style('background', '#0d1420')
      .style('color', '#e6edf3')
      .style('padding', '6px 10px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('border', '1px solid #6b7a90')
      .style('opacity', 0)
      .style('z-index', '10000')
      .style('transition', 'opacity 0.2s');

    // Node interaction handlers
    const updateHighlight = (activeNodeId: string | null) => {
      if (!activeNodeId) {
        // Reset all
        nodeSelection.style('opacity', 1);
        linkSelection.style('opacity', (d) => opacityScale(Math.abs(d.w)));
        tooltipDiv.style('opacity', 0);
        return;
      }

      const activeNode = nodeMap.get(activeNodeId);
      if (!activeNode) return;

      // Find connected neighbors
      const connectedIds = new Set<string>([activeNodeId]);
      linksCopy.forEach(({ source, target }) => {
        if (source.id === activeNodeId) {
          connectedIds.add(target.id);
        } else if (target.id === activeNodeId) {
          connectedIds.add(source.id);
        }
      });

      // Highlight connected, dim the rest
      nodeSelection.style('opacity', (d) =>
        connectedIds.has(d.id) ? 1 : 0.2
      );
      linkSelection.style('opacity', (d) => {
        if (
          connectedIds.has((d.source as SimNode).id) &&
          connectedIds.has((d.target as SimNode).id)
        ) {
          return opacityScale(Math.abs(d.w));
        }
        return 0.05;
      });

      // Show tooltip
      const layerStr =
        activeNode.layer !== null ? `layer ${activeNode.layer}` : 'logit output';
      tooltipDiv.html(
        `<strong>${layerStr}</strong><br/>feature: ${activeNode.feature}<br/>influence: ${(
          activeNode.influence * 100
        ).toFixed(1)}%`
      );
      tooltipDiv.style('opacity', 1);
    };

    nodeSelection.on('mouseenter', (event, d) => {
      setHoveredNode(d.id);
      updateHighlight(d.id);
      tooltipDiv
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY + 10) + 'px');
    });

    nodeSelection.on('mouseleave', () => {
      setHoveredNode(null);
      updateHighlight(null);
    });

    nodeSelection.on('focus', (event, d) => {
      updateHighlight(d.id);
      if (svgRef.current) {
        const rect = svgRef.current.getBoundingClientRect();
        const nodeX = d.x ?? 0;
        const nodeY = d.y ?? 0;
        tooltipDiv
          .style('left', (rect.left + nodeX + 30) + 'px')
          .style('top', (rect.top + nodeY) + 'px');
      }
    });

    nodeSelection.on('blur', () => {
      updateHighlight(null);
    });

    // Update positions on tick, or static if reduced motion
    if (prefersReducedMotion) {
      // Run simulation to completion, then stop
      simulation.tick(300);
      simulation.stop();

      nodeSelection.attr('transform', (d) => `translate(${d.x},${d.y})`);
      linkSelection
        .attr('x1', (d) => (d.source as SimNode).x ?? 0)
        .attr('y1', (d) => (d.source as SimNode).y ?? 0)
        .attr('x2', (d) => (d.target as SimNode).x ?? 0)
        .attr('y2', (d) => (d.target as SimNode).y ?? 0);
    } else {
      // Animate on tick
      simulation.on('tick', () => {
        nodeSelection.attr('transform', (d) => `translate(${d.x},${d.y})`);
        linkSelection
          .attr('x1', (d) => (d.source as SimNode).x ?? 0)
          .attr('y1', (d) => (d.source as SimNode).y ?? 0)
          .attr('x2', (d) => (d.target as SimNode).x ?? 0)
          .attr('y2', (d) => (d.target as SimNode).y ?? 0);
      });
    }

    // Cleanup on unmount
    return () => {
      simulation.stop();
      tooltipDiv.remove();
    };
  }, [width, height, prefersReducedMotion]);

  // SSR guard: render empty container on server
  if (typeof window === 'undefined') {
    return (
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: `${height}px`,
          backgroundColor: '#0a0e14',
        }}
      />
    );
  }

  return (
    <Fragment>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          backgroundColor: '#0a0e14',
        }}
      >
        <svg
          ref={svgRef}
          data-testid="neural-graph"
          role="img"
          aria-label="IOI attribution subgraph for Gemma-2-2B"
          style={{
            display: 'block',
            width: '100%',
            height: `${height}px`,
            backgroundColor: '#0a0e14',
          }}
        />

        {/* Caption with real numbers */}
        <div
          style={{
            padding: '12px 16px',
            fontSize: '13px',
            color: '#6b7a90',
            backgroundColor: '#0d1420',
            borderTop: '1px solid #1a202c',
            fontStyle: 'italic',
          }}
        >
          Slimmed from {graph.n_links_original.toLocaleString()} edges across{' '}
          {graph.n_nodes_original.toLocaleString()} nodes to the{' '}
          {graph.nodes.length} most influential features.
        </div>
      </div>

      <style>{`
        .node circle {
          transition: filter 0.15s;
        }
        .node:focus circle {
          filter: drop-shadow(0 0 6px currentColor);
        }
        .link {
          pointer-events: none;
        }
      `}</style>
    </Fragment>
  );
}
