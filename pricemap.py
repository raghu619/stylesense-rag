"""
One price axis, two rows: what qualifies, and what the search returned.

The claim this figure has to carry is that the failure is structural, not
marginal. The qualifying products occupy a narrow band at the left of the axis.
The products the search actually returned are scattered across the whole of it.

Drawing both on the SAME axis is what makes that visible. Two separate charts
would be two lists of prices and the reader would have to hold one in their head
while reading the other. Here they can point at the gap.
"""

from catalogue import parse_price_limit

GREEN, RED, INK, SOFT = "#0C6F68", "#C31E64", "#2B2733", "#6B6577"

AXIS_MAX = 10000
PLOT_L, PLOT_R = 165.0, 755.0


def _x(price: float) -> float:
    return PLOT_L + (min(price, AXIS_MAX) / AXIS_MAX) * (PLOT_R - PLOT_L)


def _dots(prices, cy: int, limit: int, spacing: float = 12.0) -> str:
    """
    One row of products, beeswarmed.

    The 13 qualifying products span Rs 799 to Rs 1,499, which is 41 pixels on a
    ten-thousand-rupee axis. Plotted on one line they merge into a blob: the
    reader sees "crowded at the left" but cannot count them, and the count is
    half the point. Stacking collided dots into lanes keeps both the density and
    the number readable, and costs nothing in accuracy since the x position of
    every dot is still its exact price.
    """
    lanes: list[list[float]] = []
    out = []
    for price in sorted(prices):
        x = _x(price)
        index = None
        for i, lane in enumerate(lanes):
            if x - lane[-1] >= spacing:
                lane.append(x)
                index = i
                break
        if index is None:
            lanes.append([x])
            index = len(lanes) - 1
        step = ((index + 1) // 2) * spacing
        offset = 0.0 if index == 0 else (step if index % 2 else -step)
        fill = GREEN if price <= limit else RED
        out.append(
            '<circle cx="%.1f" cy="%.1f" r="6" fill="%s" fill-opacity="0.9" '
            'stroke="#ffffff" stroke-width="2"/>' % (x, cy + offset, fill)
        )
    return "".join(out)


def _ticks() -> str:
    out = []
    for value in (0, 2500, 5000, 7500, 10000):
        x = _x(value)
        out.append(
            '<line x1="%.1f" y1="232" x2="%.1f" y2="238" stroke="%s" stroke-width="1"/>'
            '<text x="%.1f" y="253" text-anchor="middle" font-size="11" fill="%s">%s</text>'
            % (x, x, SOFT, x, SOFT, format(value, ","))
        )
    return "".join(out)


def render(question: str, qualify: list[int], returned: list[int]) -> str:
    """The figure. qualify and returned are plain price lists, already computed."""
    limit = parse_price_limit(question)
    if limit is None:
        return (
            '<p style="color:%s;font-size:13px">This question names no budget, '
            "so there is nothing to plot on a price axis.</p>" % SOFT
        )

    outside = sum(1 for p in returned if p > limit)
    band_l, band_r = _x(0), _x(limit)
    mid = (PLOT_L + PLOT_R) / 2

    alt = (
        "A price axis from zero to ten thousand rupees. The %d products under "
        "Rs %s sit in a narrow band at the left. Of the %d products the search "
        "returned, %d sit outside that band, scattered across the rest of the axis."
        % (len(qualify), format(limit, ","), len(returned), outside)
    )

    return """
<figure style="margin:0">
<svg viewBox="0 0 780 286" role="img" width="100%" style="max-width:780px;height:auto"
     aria-label="{alt}" font-family="system-ui, -apple-system, sans-serif">

  <rect x="{bl:.1f}" y="32" width="{bw:.1f}" height="200" fill="{green}" fill-opacity="0.16"/>
  <line x1="{br:.1f}" y1="32" x2="{br:.1f}" y2="232" stroke="{green}" stroke-width="2"/><line x1="{bl:.1f}" y1="32" x2="{bl:.1f}" y2="232" stroke="{green}" stroke-width="1" stroke-opacity="0.4"/>
  <text x="{brt:.1f}" y="24" font-size="11.5" fill="{green}" font-weight="600">the budget: Rs {limit}</text>

  <text x="152" y="80" text-anchor="end" font-size="12.5" fill="{ink}" font-weight="600">{nq} products qualify</text>
  <text x="152" y="96" text-anchor="end" font-size="11" fill="{soft}">all inside the budget</text>
  {row1}

  <text x="152" y="182" text-anchor="end" font-size="12.5" fill="{ink}" font-weight="600">{nr} results came back</text>
  <text x="152" y="198" text-anchor="end" font-size="11" fill="{red}">{outside} break the budget</text>
  {row2}

  <line x1="{pl}" y1="232" x2="{pr}" y2="232" stroke="{soft}" stroke-width="1"/>
  {ticks}
  <text x="{mid:.1f}" y="274" text-anchor="middle" font-size="11.5" fill="{soft}">price in rupees</text>
</svg>
<figcaption style="font-size:13px;color:{soft};max-width:740px;margin-top:10px">
  Both rows share one axis. The right answers sit in a narrow band. The results the
  search returned are spread across the whole of it. That is not a ranking which is
  slightly off, it is a ranking with no relationship to what was asked.
</figcaption>
</figure>""".format(
        alt=alt, bl=band_l, bw=band_r - band_l, br=band_r, brt=band_r + 7,
        limit=format(limit, ","), nq=len(qualify), nr=len(returned), outside=outside,
        row1=_dots(qualify, 88, limit), row2=_dots(returned, 190, limit),
        ticks=_ticks(), pl=PLOT_L, pr=PLOT_R, mid=mid,
        green=GREEN, red=RED, ink=INK, soft=SOFT,
    )
