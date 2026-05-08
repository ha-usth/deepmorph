from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def visualize_landmarks(
    image_path,
    point_radius=8,
    draw_index=True
):
    image_path = Path(image_path)

    # Automatically find txt file with the same filename
    txt_path = image_path.with_suffix(".txt")

    if not txt_path.exists():
        raise FileNotFoundError(f"Landmark file not found: {txt_path}")

    # Read image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try to load a default font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Read landmarks
    landmarks = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            x = float(parts[0])
            y = float(parts[1])
            landmarks.append((x, y))

    # Draw landmarks
    for idx, (x, y) in enumerate(landmarks, start=1):
        x_int = int(round(x))
        y_int = int(round(y))

        draw.ellipse(
            [
                x_int - point_radius,
                y_int - point_radius,
                x_int + point_radius,
                y_int + point_radius,
            ],
            fill="red",
            outline="white",
            width=2
        )

        if draw_index:
            draw.text(
                (x_int + point_radius + 3, y_int - point_radius - 3),
                str(idx),
                fill="yellow",
                font=font
            )

    print(f"Read landmarks from: {txt_path}")

    # Show image instead of saving
    image.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--radius", type=int, default=8, help="Point radius")
    parser.add_argument("--no-index", action="store_true", help="Do not draw landmark numbers")

    args = parser.parse_args()

    visualize_landmarks(
        image_path=args.image,
        point_radius=args.radius,
        draw_index=not args.no_index
    )