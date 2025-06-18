from PIL import Image, ImageDraw

# Grid parameters
grid_size = 12
cell_size = 40
img_size = grid_size * cell_size

# GPU green color
gpu_color = "#00FF00"

# Create base grid image (off state)
def create_grid_image():
    img = Image.new("RGB", (img_size, img_size), "black")
    draw = ImageDraw.Draw(img)
    dark_green = "#003300"
    
    for row in range(grid_size):
        for col in range(grid_size):
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            # Draw the dark green GPU tile
            draw.rectangle([x0, y0, x1, y1], fill=dark_green, outline="#006600")

            # Draw the "core" of the GPU
            padding = 6
            draw.rectangle(
                [x0 + padding, y0 + padding, x1 - padding, y1 - padding],
                fill="black"
            )
    
    return img


# Draw a single green "GPU" cell
def draw_gpu_cell(draw, x0, y0, outline_color="black"):
    x1 = x0 + cell_size
    y1 = y0 + cell_size
    draw.rectangle([x0, y0, x1, y1], fill=gpu_color, outline=outline_color)
    # Inner "core" rectangle for visual effect
    padding = 6
    draw.rectangle([x0 + padding, y0 + padding, x1 - padding, y1 - padding], fill="black")

# Full row of GPU-style green cells
def create_gpu_row_image(row_index):
    img = Image.new("RGB", (img_size, img_size), "black")
    draw = ImageDraw.Draw(img)
    for col in range(grid_size):
        x0 = col * cell_size
        y0 = row_index * cell_size
        draw_gpu_cell(draw, x0, y0)
    return img

# Full column of GPU-style green cells
def create_gpu_col_image(col_index):
    img = Image.new("RGB", (img_size, img_size), "black")
    draw = ImageDraw.Draw(img)
    for row in range(grid_size):
        x0 = col_index * cell_size
        y0 = row * cell_size
        draw_gpu_cell(draw, x0, y0)
    return img

# All rows lit with distinct colors (keep for variety)
more_colors = ["red", "green", "blue", "yellow", "orange", "cyan", "magenta", "purple"]
def create_multicolor_row_image():
    img = Image.new("RGB", (img_size, img_size), "black")
    draw = ImageDraw.Draw(img)
    for row in range(grid_size):
        color = more_colors[row % len(more_colors)]
        for col in range(grid_size):
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")
    return img

# All columns lit with distinct colors
def create_multicolor_col_image():
    img = Image.new("RGB", (img_size, img_size), "black")
    draw = ImageDraw.Draw(img)
    for col in range(grid_size):
        color = more_colors[col % len(more_colors)]
        for row in range(grid_size):
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")
    return img

# Build the animation frames
frames = []

# Red row-by-row blinking (GPU style)
for i in range(grid_size):
    frames.append(create_gpu_row_image(i))
    frames.append(create_grid_image())

# Red column-by-column blinking (GPU style)
for i in range(grid_size):
    frames.append(create_gpu_col_image(i))
    frames.append(create_grid_image())

# Add multicolor effects (just for flair at the end)
frames.append(create_multicolor_row_image())
frames.append(create_grid_image())
frames.append(create_multicolor_col_image())
frames.append(create_grid_image())


# Build matching durations list (faster for GPU blinks, slower for color flashes)
durations = []

# 8 rows and 8 cols → 16 GPU blink pairs = 32 frames (75ms each)
durations += [75] * (2 * grid_size)  # Red rows
durations += [75] * (2 * grid_size)  # Red cols

# Add slower durations for multicolor row/col highlights
durations += [400, 100]  # Multicolor rows: show then off
durations += [400, 100]  # Multicolor cols: show then off



# Save GIF: 4x faster (75ms per frame)
gif_path_full = "dl_comm_logo.gif"
frames[0].save(
    gif_path_full,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0
)

print(f"GIF saved to: {gif_path_full}")
