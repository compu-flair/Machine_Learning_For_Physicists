# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D

# plt.style.use('dark_background')  # Dark theme

# # Define loss function and gradient
# def loss(w1, w2):
#     return (w1 - 3)**2 + (w2 + 2)**2

# def gradient(w1, w2):
#     return np.array([2 * (w1 - 3), 2 * (w2 + 2)])

# # Simulate gradient descent
# w = np.array([0.0, 0.0])
# learning_rate = 0.1
# points = [w.copy()]
# for _ in range(50):
#     grad = gradient(*w)
#     w = w - learning_rate * grad
#     points.append(w.copy())

# points = np.array(points)

# # Loss surface grid
# W1, W2 = np.meshgrid(np.linspace(-1, 6, 100), np.linspace(-5, 2, 100))
# L = loss(W1, W2)

# # Create 3D figure
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Apply double pendulum styling
# ax.grid(False)
# ax.xaxis.pane.set_visible(False)
# ax.yaxis.pane.set_visible(False)
# ax.zaxis.pane.set_visible(False)
# ax.set_facecolor("black")

# # Plot surface
# ax.plot_surface(W1, W2, L, cmap='viridis', alpha=0.7)

# # Labels
# ax.set_xlabel('w1')
# ax.set_ylabel('w2')
# ax.set_zlabel('Loss')
# ax.set_xlim([-1, 6])
# ax.set_ylim([-5, 2])
# ax.set_zlim([0, 50])
# ax.set_title('Loss function minimization')

# # Descent path
# path, = ax.plot([], [], [], 'r.-', markersize=5, label='Descent Path')
# ax.legend()

# # Update function for animation
# def update(frame):
#     path.set_data(points[:frame+1, 0], points[:frame+1, 1])
#     path.set_3d_properties(loss(points[:frame+1, 0], points[:frame+1, 1]))
#     return path,

# # Animate
# ani = FuncAnimation(fig, update, frames=len(points), interval=100, blit=False)
# ani.save("loss_function_minimization.mp4", writer="ffmpeg", fps=30, dpi=500)

import math
import pandas as pd
from manim import *
import os 

class DataPoint(Dot):
    def __init__(self, point: list | np.ndarray, x: float, y: float, color, **kwargs):
        super().__init__(point=point, radius=.15, color=color, **kwargs)
        self.x = x
        self.y = y

def create_model(data: pd.DataFrame, initial_m: float, initial_b: float) -> tuple:

    m_tracker = ValueTracker(initial_m)
    b_tracker = ValueTracker(initial_b)

    ax = Axes(
        x_range=[-0.5, 10],
        y_range=[-0.2, 1.3],
        x_axis_config={"include_tip": False, "include_numbers": False},
        y_axis_config={"include_tip": False, "include_numbers": True, "stroke_opacity": 0}
    )

    # plot points
    false_points = [DataPoint(point=ax.c2p(p.x, p.y), x=p.x, y=p.y, color=RED) for p in data.itertuples() if p.y == 0.0]
    true_points = [DataPoint(point=ax.c2p(p.x, p.y), x=p.x, y=p.y, color=BLUE) for p in data.itertuples() if p.y == 1.0]
    points = [*true_points, *false_points]

    # plot function
    f = lambda x: 1.0 / (1.0 + math.exp(-(b_tracker.get_value() + m_tracker.get_value() * x)))
    plot = always_redraw(lambda: ax.plot(f, color=YELLOW))

    # max line
    max_line = DashedLine(start=ax.c2p(0, 1), end=ax.c2p(10, 1), color=WHITE)

    # likelihood_lines
    likelihood_lines = [
        always_redraw(
            lambda p=p: DashedLine(
                start=p.get_center(),
                end=ax.c2p(p.x, f(p.x)),
                color=p.get_color()
            )
        )
        for p in points
    ]

    return data, m_tracker, b_tracker, ax, points, true_points, false_points, plot, f, max_line, likelihood_lines

# class LogisticRegressionScene(Scene):

#     def construct(self):

#         # build the logistic regression model
#         url = r"https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/simple_logistic_regression.csv"

#         data, m_tracker, b_tracker, ax, points, true_points, false_points, \
#             plot, f, max_line, likelihood_lines = create_model(data=pd.read_csv(url),
#                                                                initial_m=0.69267212,
#                                                                initial_b=-3.17576395
#                                                                )

#         # draw and initialize the objects
#         self.play(LaggedStartMap(Write, ax),
#                   Write(max_line),
#                   Write(MathTex("0") \
#                         .scale(.8) \
#                         .next_to(ax.c2p(0, 0), DL, buff=.2)
#                         )
#                   )

#         self.play(LaggedStartMap(Write, VGroup(*true_points)))
#         self.play(LaggedStartMap(Write, VGroup(*false_points)))
#         self.play(Write(plot))

#         # draw axis labels
#         x_label = ax.get_x_axis_label(
#             Text("Temperature (°C)").scale(0.4), edge=DOWN, direction=DOWN, buff=0.5
#         )
#         y_label = ax.get_y_axis_label(
#             Text("Ice Cream Sales").scale(0.4).rotate(90 * DEGREES),
#             edge=LEFT,
#             direction=LEFT,
#             buff=0.3,
#         )

#         title = Text("Probabilistic Model(s)", font="Orbitron", color=YELLOW).shift(UP*2.5)
#         self.play(Write(x_label),Write(y_label), DrawBorderThenFill(title))
#         self.play(LaggedStartMap(Write, VGroup(*likelihood_lines)))
#         self.wait()




# class LogisticRegressionScene1(Scene):

#     def construct(self):

#         # build the logistic regression model
#         url = r"https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/simple_logistic_regression.csv"

#         data, m_tracker, b_tracker, ax, points, true_points, false_points, \
#             plot, f, max_line, likelihood_lines = create_model(data=pd.read_csv(url),
#                                                                initial_m=0.69267212,
#                                                                initial_b=-3.17576395
#                                                                )

#         # draw and initialize the objects
#         self.play(LaggedStartMap(Write, ax),
#                   Write(max_line),
#                   Write(MathTex("0") \
#                         .scale(.8) \
#                         .next_to(ax.c2p(0, 0), DL, buff=.2)
#                         )
#                   )

#         self.play(LaggedStartMap(Write, VGroup(*true_points)))
#         self.play(LaggedStartMap(Write, VGroup(*false_points)))
#         self.play(Write(plot))

#         # draw axis labels
#         x_label = ax.get_x_axis_label(
#             Text("Temperature (°C)").scale(0.4), edge=DOWN, direction=DOWN, buff=0.5
#         )
#         y_label = ax.get_y_axis_label(
#             Text("Ice Cream Sales").scale(0.4).rotate(90 * DEGREES),
#             edge=LEFT,
#             direction=LEFT,
#             buff=0.3,
#         )

#         self.play(Write(x_label),Write(y_label))
#         self.play(LaggedStartMap(Write, VGroup(*likelihood_lines)))
#         self.wait()


# from manim import *
# from PIL import Image
# import os

# class SplitImage(Scene):
#     def construct(self):
#         image_path = "car.png"
#         full_image = ImageMobject(image_path).scale(0.3).shift(LEFT*3)
#         self.play(FadeIn(full_image))
#         original = Image.open(image_path)
#         width, height = original.size

#         rows, cols = 3, 3  # Grid size
#         tile_width = width // cols
#         tile_height = height // rows

#         tiles = Group()
#         for i in range(rows):
#             for j in range(cols):
#                 # Crop tile
#                 left = j * tile_width
#                 upper = i * tile_height
#                 right = left + tile_width
#                 lower = upper + tile_height
#                 tile_img = original.crop((left, upper, right, lower))

#                 # Save tile to a temporary file
#                 tile_filename = f"tile_{i}_{j}.png"
#                 tile_img.save(tile_filename)

#                 # Load in Manim
#                 tile = ImageMobject(tile_filename).scale(0.5 / max(rows, cols))
#                 tile.move_to(
#                     RIGHT*1.2* (j - (cols - 1) / 2) +
#                     DOWN *1.2* (i - (rows - 1) / 2)
#                 )
#                 tiles.add(tile)
#         tiles.shift(RIGHT*3)
#         arrow = Arrow(full_image.get_right(), tiles.get_left(), color=YELLOW, tip_length=0.2).scale([0.7,1,1])
#         self.play(GrowArrow(arrow))
#         self.play(*[FadeIn(tile) for tile in tiles])
#         title = Text("Feature Extraction", color=YELLOW).shift(UP*2)
#         self.play(DrawBorderThenFill(title))
#         self.wait()

#         # Optional cleanup
#         for i in range(rows):
#             for j in range(cols):
#                 os.remove(f"tile_{i}_{j}.png")

# from manim import *

# class Main(Scene):
#     def construct(self):
#         # Create two large circles
#         circle1 = Circle(radius=2.5, color=YELLOW).set_fill(YELLOW, opacity=0.2).shift(LEFT * 3.5)
#         circle2 = Circle(radius=2.5, color=TEAL).set_fill(TEAL, opacity=0.2).shift(RIGHT * 3.5)
#         self.play(Create(circle1), Create(circle2))

#         # Add labels to circles
#         label1 = Text("Machine Learning", font="Orbitron", color=YELLOW).scale(0.5).next_to(circle1, UP)
#         label2 = Text("Physics", font="Orbitron", color=TEAL).scale(0.5).next_to(circle2, UP)
#         self.play(Write(label1), Write(label2))

#         tip_style = dict(
#             tip_shape_start=StealthTip,
#             tip_shape_end=StealthTip
#         )
#         arrow1 = CurvedDoubleArrow(
#             circle1.get_center() + UP * 2 + RIGHT,
#             circle2.get_center() + UP * 2 + LEFT,
#             radius=-5,**tip_style
#         ).set_stroke(width=4)

#         arrow2 = DoubleArrow(
#             circle1.get_center() + RIGHT,
#             circle2.get_center() + LEFT,**tip_style
#         ).set_stroke(width=4)

#         arrow3 = CurvedDoubleArrow(
#             circle1.get_center() + DOWN * 2 + RIGHT,
#             circle2.get_center() + DOWN * 2 + LEFT,
#             radius=5,**tip_style
#         ).set_stroke(width=4)

#         self.play(Create(arrow1), Create(arrow2), Create(arrow3))
#         self.wait(5)
#         self.play(FadeOut(*self.mobjects))
#         ax = Axes(
#             x_range=[0, 10, 1],
#             y_range=[0, 10, 1],
#             axis_config={"include_numbers": True},
#         ).scale(0.7)
#         self.play(Create(ax))

#         # Generate synthetic data
#         np.random.seed(42)
#         x_data = np.linspace(1, 9, 15)
#         true_slope = 0.8
#         true_intercept = 1.5
#         y_data = true_slope * x_data + true_intercept + np.random.normal(0, 1.5, size=x_data.shape)

#         # Show data points
#         points = VGroup(*[
#             Dot(ax.c2p(x, y), color=YELLOW) for x, y in zip(x_data, y_data)
#         ])
#         self.play(FadeIn(points, lag_ratio=0.1))
#         self.wait()

#         # Set up slope and intercept as trackers
#         slope_tracker = ValueTracker(-0.5)     # fake initial slope
#         intercept_tracker = ValueTracker(9.0)  # fake initial intercept

#         # Dynamic regression line using always_redraw
#         def get_line():
#             x_min, x_max = min(x_data), max(x_data)
#             y_start = slope_tracker.get_value() * x_min + intercept_tracker.get_value()
#             y_end = slope_tracker.get_value() * x_max + intercept_tracker.get_value()
#             return ax.plot_line_graph(
#                 x_values=[x_min, x_max],
#                 y_values=[y_start, y_end],
#                 line_color=TEAL,
#                 stroke_width=2
#             )

#         regression_line = always_redraw(get_line)
#         self.play(FadeIn(regression_line))
#         self.wait()

#         # Dotted vertical error lines from data points to regression line (also always updating)
#         error_lines = VGroup()
#         for x, y in zip(x_data, y_data):
#             err_line = always_redraw(lambda x_=x, y_=y: DashedLine(
#                 start=ax.c2p(x_, y_),
#                 end=ax.c2p(x_, slope_tracker.get_value() * x_ + intercept_tracker.get_value()),
#                 color=PURE_GREEN,
#                 stroke_opacity=0.6
#             ))
#             error_lines.add(err_line)
#         self.add(error_lines)

#         # Animate slope and intercept to true values
#         self.play(
#             slope_tracker.animate.set_value(true_slope),
#             intercept_tracker.animate.set_value(true_intercept),
#             run_time=3
#         )

#         self.wait()
#         for obj in self.mobjects:
#             obj.clear_updaters()
#         graph_group = Group(*self.mobjects)
#         self.play(graph_group.animate.scale(0.5).shift(UP*1.5+LEFT*3))
#         arrow1 = Arrow(LEFT,RIGHT, tip_length=0.2, stroke_width=3, color=YELLOW).scale([0.8,1,1]).next_to(graph_group, RIGHT)
#         arrow2 = Arrow(UP,DOWN, tip_length=0.2, stroke_width=3, color=YELLOW).scale([1,0.5,1]).next_to(graph_group, DOWN , aligned_edge=LEFT)
#         arrow3 = Arrow(UP,DOWN, tip_length=0.2, stroke_width=3, color=YELLOW).scale([1,0.5,1]).next_to(graph_group, DOWN , aligned_edge=RIGHT)
#         self.play(GrowArrow(arrow1),GrowArrow(arrow2),GrowArrow(arrow3))
#         logistic_regression = Text("Logistic Regression", font="Orbitron", color=TEAL).scale(0.3).next_to(arrow1, RIGHT)
#         neural_network = Text("Neural Networks", font="Orbitron", color=TEAL).scale(0.3).next_to(arrow2, DOWN)
#         svm = Text("Support Vector Machines", font="Orbitron", color=TEAL).scale(0.3).next_to(arrow3, DOWN)
#         self.play(FadeIn(logistic_regression,neural_network,svm))
#         self.wait()



# from manim import *
# import numpy as np
# import random

# class LeastActionMixedPaths(Scene):
#     def construct(self):
#         # Title
#         title = Text("Least Action Principle", font="Orbitron", color=YELLOW).shift(UP*3)
#         self.play(Write(title))

#         # Define start and end points
#         A = LEFT * 5 + DOWN * 2
#         B = RIGHT * 5 + DOWN * 2
#         dot_A = Dot(A, color=YELLOW)
#         dot_B = Dot(B, color=YELLOW)
#         self.play(FadeIn(dot_A), FadeIn(dot_B))

#         # Label points A and B
#         label_A = Text("A", font_size=30).next_to(dot_A, LEFT)
#         label_B = Text("B", font_size=30).next_to(dot_B, RIGHT)
#         self.play(Write(label_A), Write(label_B))

#         # Optimal straight path
#         optimal_path = Line(A, B).set_color(PURE_GREEN).set_stroke(width=2)

#         all_paths = VGroup()

#         # --- Wavy paths ---
#         for i in range(2):
#             x_vals = np.linspace(0, 1, 60)
#             y_scale = 0.3 + i * 0.2
#             points = []
#             for alpha in x_vals:
#                 x = interpolate(A[0], B[0], alpha)
#                 y_wave = np.sin(2 * np.pi * alpha * (3 + i)) * y_scale
#                 y = interpolate(A[1], B[1], alpha) + y_wave
#                 points.append(np.array([x, y, 0]))
#             path = VMobject().set_color(RED).set_stroke(width=2)
#             path.set_points_smoothly(points)
#             all_paths.add(path)

#         # --- Arc-like Bezier paths ---
#         for dy in [3.5, -3]:
#             control1 = A + UP * dy + RIGHT
#             control2 = B + UP * dy + LEFT
#             arc_path = CubicBezier(A, control1, control2, B).set_color(RED).set_stroke(width=2)
#             all_paths.add(arc_path)

#         # --- Jittery path ---
#         jitter_points = []
#         num_points = 80
#         for i in range(num_points):
#             alpha = i / (num_points - 1)
#             x = interpolate(A[0], B[0], alpha)
#             base_y = interpolate(A[1], B[1], alpha)
#             if i == 0 or i == num_points - 1:
#                 y = base_y
#             else:
#                 y = base_y + random.uniform(-0.7, 0.7)
#             jitter_points.append(np.array([x, y, 0]))
#         jitter_path = VMobject().set_color(RED).set_stroke(width=2)
#         jitter_path.set_points_as_corners(jitter_points)
#         all_paths.add(jitter_path)

#         self.play(Create(all_paths), run_time=2)
#         self.wait(0.5)

#         self.play(Create(optimal_path), run_time=1.5)

#         # Particle follows optimal path
#         particle = Dot(color=BLUE).move_to(A)
#         self.add(particle)
#         self.play(MoveAlongPath(particle, optimal_path, rate_func=linear, run_time=3))

#         # === Legend ===
#         legend = VGroup(
#             # Blue particle
#             VGroup(Dot(color=BLUE), Text("Particle", font_size=26)).arrange(RIGHT, buff=0.3),
#             # Green path
#             VGroup(Line(ORIGIN, RIGHT*0.8).set_color(PURE_GREEN), Text("Optimal Path", font_size=26)).arrange(RIGHT, buff=0.3),
#             # Red paths
#             VGroup(Line(ORIGIN, RIGHT*0.8).set_color(RED), Text("Non-optimal Paths", font_size=26)).arrange(RIGHT, buff=0.3),
#         ).arrange(DOWN, aligned_edge=LEFT).shift(RIGHT*4+UP*1.5)

#         self.play(FadeIn(legend))
#         self.wait()

# from manim import *

# class SquareFunction(Scene):
# 	def construct(self):
# 		axes = Axes(
#             x_range = [-5, 5.1, 1],
#             y_range = [-1.5, 1.5, 1],
#             x_length = 2*TAU,
#             axis_config = {"color": PURE_GREEN},
#             # x_axis_config = {
#             # "numbers_to_include":np.arange(-10, 10.1, 2),
#             # # "numbers_with_elongated_ticks": np.arange(-10, 10.2, 2),
#             # },
#             tips = False,
#             ).scale(0.7)

# 		def squareWave(x):
# 			if int(x)%2 == 0:
# 				return 1*x//abs(x)
# 			else:
# 				return -1*x//abs(x)
# 		# def Sin1(x):
# 		# 	return 2*(1-np.cos(1))*np.sin(x)
# 		# def Sin2(x):
# 		# 	return (2/2)*(1-np.cos(2))*np.sin(2*x)

# 		def Sinn(x, n):
# 			result = 0
# 			for i in range(1, n+1, 2):
# 				result += (2/(i*np.pi))*(2)*np.sin(i*x*np.pi)
# 			return result
# 		def SINN(x, i):
# 			return (2/(i*np.pi))*(2)*np.sin(i*x*np.pi)


# 		square_graph = axes.plot(squareWave, x_range = (-4.9, 4.9, 0.01), **{"discontinuities": [x for x in range(-5, 6)]})
# 		values = [0]
# 		values1 = [0]
# 		index = 1
# 		for i in range(1, 31, 2):
# 			values.append(axes.plot(lambda x: SINN(x, i), x_range = (-4.9, 4.9, 0.1)))
# 			values1.append(axes.plot(lambda x: Sinn(x, i), x_range = (-4.9, 4.9, 0.1), color = TEAL))
# 		#sin1_graph = axes.plot(Sin1, x_range = (-4.9, 4.9, 0.01))
# 		#sin2_graph = axes.plot(Sin2, x_range = (-4.9, 4.9, 0.01))
# 		#sinn_graph = axes.plot(Sinn, x_range = (-4.9, 4.9, 0.1))
# 		self.add(axes, square_graph,  values[1])
# 		title = Text("Fourier Transform", font="Orbitron", color=YELLOW).shift(UP*3)
# 		self.play(DrawBorderThenFill(title))
# 		self.play(ReplacementTransform(values[1], values1[1]))
# 		for i in range(2, 5):
# 			self.play(Create(values[i]))
# 			self.wait(1)
# 			self.play(ReplacementTransform(values[i], values1[i]), ReplacementTransform(values1[i-1], values1[i]))



# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from mpl_toolkits.mplot3d import Axes3D

# # Use dark background style
# plt.style.use("dark_background")

# # Set up figure and 3D axis
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Turn off grid and axis panes (like your pendulum plot)
# ax.grid(False)
# ax.xaxis.pane.set_visible(False)
# ax.yaxis.pane.set_visible(False)
# ax.zaxis.pane.set_visible(False)
# ax.set_facecolor("black")

# # Generate synthetic data
# np.random.seed(0)
# class1 = np.random.normal(loc=[-2, -2, 2], scale=0.5, size=(8, 3))
# class2 = np.random.normal(loc=[2, 2, -1], scale=0.5, size=(8, 3))

# # Plot the points
# ax.scatter(*class1.T, color='blue', s=50, label='Class +1')
# ax.scatter(*class2.T, color='red', s=50, label='Class -1')

# # Define a tilted plane: ax + by + cz + d = 0
# a, b, c, d = 1, 1, -1, 0

# # Smaller meshgrid to control plane size
# xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
# zz = (-a * xx - b * yy - d) / c

# # Plot the decision plane
# ax.plot_surface(xx, yy, zz, alpha=0.4, color='white', edgecolor='gray')

# # Support vectors
# sv1 = class1[0]
# sv2 = class2[0]
# ax.scatter(*sv1, color='yellow', s=100, edgecolor='black', linewidth=1.5, label='Support Vectors')
# ax.scatter(*sv2, color='yellow', s=100, edgecolor='black', linewidth=1.5)

# # Set axis limits and labels
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)
# ax.set_zlim(-2, 3)
# ax.set_xlabel("X Position")
# ax.set_ylabel("Y Position")
# ax.set_zlabel("Z Position")

# # Legend
# ax.legend()

# # Optional camera rotation animation
# def rotate(angle):
#     ax.view_init(elev=25, azim=angle)

# ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)

# ani.save("svm_plane_styled.mp4", writer="ffmpeg", dpi=500, fps=20)

# plt.show()



# from manim import *
# import numpy as np

# class Cylinder(VGroup):
#     def __init__(self, radius=1, height=2, color=BLUE, num_lines=800, fill_circles=False, fill_opacity=0.3, **kwargs):
#         super().__init__(**kwargs)
#         self.radius = radius
#         self.height = height
        
#         # Top and bottom circle outlines or fills
#         if fill_circles:
#             top_circle = Circle(radius=radius, color=color, fill_opacity=fill_opacity).set_stroke(color=color, opacity=fill_opacity).move_to([0, 0, height / 2])
#             bottom_circle = Circle(radius=radius, color=color, fill_opacity=fill_opacity).set_stroke(color=color, opacity=fill_opacity).move_to([0, 0, -height / 2])
#         else:
#             top_circle = ParametricFunction(
#                 lambda t: np.array([radius * np.cos(t), radius * np.sin(t), height / 2]),
#                 t_range=[0, TAU],
#                 color=color,
#                 stroke_width=2
#             )
#             bottom_circle = ParametricFunction(
#                 lambda t: np.array([radius * np.cos(t), radius * np.sin(t), -height / 2]),
#                 t_range=[0, TAU],
#                 color=color,
#                 stroke_width=2
#             )

#         # Vertical curved surface (connecting lines)
#         vertical_surface = VGroup(*[
#             ParametricFunction(
#                 lambda t: np.array([
#                     radius * np.cos(theta),
#                     radius * np.sin(theta),
#                     t
#                 ]),
#                 t_range=[-height / 2, height / 2],
#                 color=color,
#                 stroke_width=1
#             ).set_opacity(fill_opacity) for theta in np.linspace(0, TAU, num_lines, endpoint=False)  # Avoid duplicate line
#         ])

#         self.add(top_circle, bottom_circle, vertical_surface)

# class MovingParticles(VGroup):
#     def __init__(self, num_particles=20, radius=0.02, container=None, **kwargs):
#         super().__init__(**kwargs)
#         self.container = container
#         self.particles = VGroup()
#         self.velocities = []
#         self.speed_factor = 1  # Start with a speed factor of 1

#         for _ in range(num_particles):
#             x, y, z = np.random.uniform(-container.radius, container.radius), np.random.uniform(-container.radius, container.radius), np.random.uniform(-container.height/2, container.height / 9.5)
#             particle = Dot3D(radius=radius, color=YELLOW).move_to([x, y, z])
#             self.particles.add(particle)
#             self.velocities.append(np.random.uniform(-0.1, 0.1, 3))
        
#         self.add(self.particles)
#         self.add_updater(self.update_particles)
    
#     def update_particles(self, mob, dt):
#         self.speed_factor += 0.02 * dt  # Gradually increase speed over time
        
#         for i, particle in enumerate(self.particles):
#             velocity = self.velocities[i]   # Apply speed factor
#             new_position = particle.get_center() + velocity * dt * 30
#             x, y, z = new_position

#             # Ensure particles stay inside the cylinder
#             if np.linalg.norm([x, y]) > self.container.radius:
#                 normal = np.array([x, y]) / np.linalg.norm([x, y])  
#                 velocity[:2] -= 2 * np.dot(velocity[:2], normal) * normal  
#                 x, y = self.container.radius * normal  

#             if abs(z) > self.container.height / 9.5:
#                 velocity[2] *= -1
#                 z = np.clip(z, -self.container.height, self.container.height / 9.5)

#             # Update velocity and position
#             self.velocities[i] = velocity
#             particle.move_to([x, y, z])


# class ParametricCylinderScene(ThreeDScene):
#     def construct(self):
#         self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
#         cylinder = Cylinder().scale(1.5)
#         liquid = Cylinder(radius=0.99, height=1.2, color=WHITE, fill_circles=True).scale(1.5).move_to(cylinder, aligned_edge=IN)
#         particles = MovingParticles(num_particles=30, container=liquid)
#         liquid1 = Cylinder(radius=0.99, height=0.3, color=WHITE, fill_circles=True).scale(1.5).move_to(cylinder, aligned_edge=IN)
#         self.wait(3)
#         self.play(Create(cylinder))
#         self.play(FadeIn(liquid), Create(particles))
#         self.play(liquid.animate.become(liquid1), run_time=8, rate_func=rate_functions.linear)
#         self.wait(1)


# from manim import *
# import cv2
# import os

# class CanonicalEnsembleSnapshots(Scene):
#     def construct(self):
#         video_path = r"C:\Users\iT HOME\Desktop\media\videos\exp\480p15\ParametricCylinderScene.mp4"  # Path to your video
#         snapshot_dir = "snapshots"
#         os.makedirs(snapshot_dir, exist_ok=True)

#         # Step 1: Capture frames using OpenCV
#         cap = cv2.VideoCapture(video_path)
#         frame_rate = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         duration = total_frames / frame_rate

#         snapshot_times = [4.5,5.1,5.7,6, 6.5, 7.1, 7.3, 7.8 , 8.3 , 8.9 , 9.2, 9.5]  # seconds to sample
#         snapshot_paths = []

#         for i, t in enumerate(snapshot_times):
#             frame_num = int(t * frame_rate)
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#             success, frame = cap.read()
#             if success:
#                 img_path = f"{snapshot_dir}/snapshot_{i}.png"
#                 cv2.imwrite(img_path, frame)
#                 snapshot_paths.append(img_path)

#         cap.release()

#         snapshots = []
#         for i, img_path in enumerate(snapshot_paths):
#             img = ImageMobject(img_path).scale(0.6)
#             label = Text(f"t = {snapshot_times[i]}s", font="Orbitron", color=YELLOW).scale(0.3)
#             group = Group(img, label).arrange(DOWN*0.5)
#             snapshots.append(group)

#         # --- Step 3: Arrange all in a grid ---
#         grid = Group(*snapshots).arrange_in_grid(rows=4, cols=4, buff=(0,0)).scale(0.9)

#         for i, group in enumerate(snapshots):
#             self.play(FadeIn(group), run_time=0.3)
#             self.wait(0.1)
#         self.wait(2)

