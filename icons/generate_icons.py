"""Generate plugin icons for reverb, fractal, and lossy.

Each icon is 256x256 PNG with a distinctive visual identity matching the
plugin's color theme. Run once to regenerate all icons.
"""

import math
from PIL import Image, ImageDraw


SIZE = 256
CENTER = SIZE // 2


def reverb_icon():
    """Blue/violet concentric ripples — sound waves reverberating in space."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw arcs on a separate layer, then mask to circle
    bg = Image.new("RGBA", (SIZE, SIZE), (12, 8, 28, 255))
    arc_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    arc_draw = ImageDraw.Draw(arc_layer)

    # Concentric ripple arcs emanating from lower-left
    ox, oy = SIZE * 0.35, SIZE * 0.65
    radii = [28, 52, 76, 100, 124]
    for i, r in enumerate(radii):
        t = i / (len(radii) - 1)
        cr = int(100 + 120 * (1 - t))
        cg = int(40 + 40 * (1 - t))
        cb = int(200 + 55 * (1 - t))
        alpha = int(255 * (1 - 0.35 * t))
        width = max(3, int(5 - t * 2))
        bbox = [ox - r, oy - r, ox + r, oy + r]
        arc_draw.arc(bbox, 210, 360, fill=(cr, cg, cb, alpha), width=width)
        arc_draw.arc(bbox, 0, 30, fill=(cr, cg, cb, alpha), width=width)

    # Bright source dot
    arc_draw.ellipse([ox - 6, oy - 6, ox + 6, oy + 6], fill=(200, 160, 255, 255))

    # Circular mask
    circle_mask = Image.new("L", (SIZE, SIZE), 0)
    cm_draw = ImageDraw.Draw(circle_mask)
    cm_draw.ellipse([6, 6, SIZE - 7, SIZE - 7], fill=255)

    # Composite: bg circle + masked arcs
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    bg.putalpha(circle_mask)
    img = Image.alpha_composite(img, bg)
    arc_layer.putalpha(circle_mask)
    img = Image.alpha_composite(img, arc_layer)

    # Subtle outer ring
    draw = ImageDraw.Draw(img)
    draw.ellipse([6, 6, SIZE - 7, SIZE - 7], outline=(100, 60, 200, 80), width=2)

    return img


def fractal_icon():
    """Amber recursive geometric pattern — Sierpinski-like triangles."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark amber background circle
    draw.ellipse([4, 4, SIZE - 5, SIZE - 5], fill=(18, 12, 6, 255))

    def sierpinski(ax, ay, bx, by, cx, cy, depth, max_depth):
        if depth >= max_depth:
            return
        t = depth / max(max_depth - 1, 1)
        # Bright amber at shallow, dim at deep
        r = int(255 - 80 * t)
        g = int(180 - 80 * t)
        b = int(0 + 20 * t)
        alpha = int(255 - 100 * t)
        w = max(2, int(3 - t))
        draw.polygon([(ax, ay), (bx, by), (cx, cy)],
                      outline=(r, g, b, alpha), width=w)

        # Three sub-triangles
        mx_ab, my_ab = (ax + bx) / 2, (ay + by) / 2
        mx_bc, my_bc = (bx + cx) / 2, (by + cy) / 2
        mx_ca, my_ca = (cx + ax) / 2, (cy + ay) / 2
        sierpinski(ax, ay, mx_ab, my_ab, mx_ca, my_ca, depth + 1, max_depth)
        sierpinski(mx_ab, my_ab, bx, by, mx_bc, my_bc, depth + 1, max_depth)
        sierpinski(mx_ca, my_ca, mx_bc, my_bc, cx, cy, depth + 1, max_depth)

    # Main triangle centered in the circle
    pad = 42
    top = (CENTER, pad + 10)
    bl = (pad, SIZE - pad - 10)
    br = (SIZE - pad, SIZE - pad - 10)
    sierpinski(top[0], top[1], bl[0], bl[1], br[0], br[1], 0, 4)

    # Subtle outer ring
    draw.ellipse([6, 6, SIZE - 7, SIZE - 7], outline=(120, 80, 20, 80), width=2)

    return img


def lossy_icon():
    """Green glitch blocks — data corruption / codec artifacts."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark green background circle
    draw.ellipse([4, 4, SIZE - 5, SIZE - 5], fill=(8, 18, 10, 255))

    # Create a mask so blocks only appear inside the circle
    mask = Image.new("L", (SIZE, SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([12, 12, SIZE - 13, SIZE - 13], fill=255)

    # Glitch blocks — a waveform that gets progressively corrupted
    import random
    rng = random.Random(42)  # deterministic

    block_size = 12
    cols = SIZE // block_size
    rows = SIZE // block_size

    for row in range(rows):
        for col in range(cols):
            x = col * block_size
            y = row * block_size
            cx = x + block_size / 2
            cy = y + block_size / 2

            # Distance from center
            dx = (cx - CENTER) / CENTER
            dy = (cy - CENTER) / CENTER
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0.85:
                continue

            # Sine wave base — brighter near center band
            wave = math.sin(cx * 0.08) * 0.4
            band_dist = abs(dy - wave * 0.3)

            # Corruption increases toward edges
            corrupt = rng.random() < (0.15 + 0.5 * dist)

            if band_dist < 0.25:
                # Signal band
                intensity = max(0, 1.0 - band_dist * 4)
                if corrupt:
                    # Glitched block — shifted green
                    g = int(80 + 175 * rng.random())
                    r = int(20 * rng.random())
                    b = int(30 * rng.random())
                else:
                    g = int(60 + 195 * intensity)
                    r = 0
                    b = int(20 * intensity)
                alpha = int(180 + 75 * intensity)
                draw.rectangle([x + 1, y + 1, x + block_size - 1, y + block_size - 1],
                               fill=(r, g, b, alpha))
            elif corrupt and dist < 0.75:
                # Scattered artifact blocks
                g = int(30 + 50 * rng.random())
                draw.rectangle([x + 1, y + 1, x + block_size - 1, y + block_size - 1],
                               fill=(0, g, int(g * 0.3), 60))

    # Apply circular mask
    img.putalpha(mask)

    # Re-draw outer ring on top
    draw2 = ImageDraw.Draw(img)
    draw2.ellipse([6, 6, SIZE - 7, SIZE - 7], outline=(0, 102, 48, 80), width=2)

    return img


if __name__ == "__main__":
    import os
    out = os.path.dirname(os.path.abspath(__file__))

    icons = {
        "reverb": reverb_icon(),
        "fractal": fractal_icon(),
        "lossy": lossy_icon(),
    }

    for name, img in icons.items():
        png_path = os.path.join(out, f"{name}.png")
        img.save(png_path)
        print(f"  {png_path}")

        # Also save .ico (Windows) — 256, 64, 32, 16
        ico_path = os.path.join(out, f"{name}.ico")
        sizes = [(256, 256), (64, 64), (48, 48), (32, 32), (16, 16)]
        img.save(ico_path, format="ICO",
                 sizes=sizes)
        print(f"  {ico_path}")

    print("Done.")
