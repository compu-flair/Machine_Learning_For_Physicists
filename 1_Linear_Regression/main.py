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
# ax.set_title('Loss function minimization', fontsize=22, color='yellow')

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

# import math
# import pandas as pd
# from manim import *
# import os 

# class DataPoint(Dot):
#     def __init__(self, point: list | np.ndarray, x: float, y: float, color, **kwargs):
#         super().__init__(point=point, radius=.15, color=color, **kwargs)
#         self.x = x
#         self.y = y

# def create_model(data: pd.DataFrame, initial_m: float, initial_b: float) -> tuple:

#     m_tracker = ValueTracker(initial_m)
#     b_tracker = ValueTracker(initial_b)

#     ax = Axes(
#         x_range=[-0.5, 10],
#         y_range=[-0.2, 1.3],
#         x_axis_config={"include_tip": False, "include_numbers": False},
#         y_axis_config={"include_tip": False, "include_numbers": True, "stroke_opacity": 0}
#     )

#     # plot points
#     false_points = [DataPoint(point=ax.c2p(p.x, p.y), x=p.x, y=p.y, color=RED) for p in data.itertuples() if p.y == 0.0]
#     true_points = [DataPoint(point=ax.c2p(p.x, p.y), x=p.x, y=p.y, color=BLUE) for p in data.itertuples() if p.y == 1.0]
#     points = [*true_points, *false_points]

#     # plot function
#     f = lambda x: 1.0 / (1.0 + math.exp(-(b_tracker.get_value() + m_tracker.get_value() * x)))
#     plot = always_redraw(lambda: ax.plot(f, color=YELLOW))

#     # max line
#     max_line = DashedLine(start=ax.c2p(0, 1), end=ax.c2p(10, 1), color=WHITE)

#     # likelihood_lines
#     likelihood_lines = [
#         always_redraw(
#             lambda p=p: DashedLine(
#                 start=p.get_center(),
#                 end=ax.c2p(p.x, f(p.x)),
#                 color=p.get_color()
#             )
#         )
#         for p in points
#     ]

#     return data, m_tracker, b_tracker, ax, points, true_points, false_points, plot, f, max_line, likelihood_lines

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
#         title = Text("Feature Extraction", font="Orbitron" , color=YELLOW).shift(UP*2)
#         self.play(DrawBorderThenFill(title))
#         self.wait()

#         # Optional cleanup
#         for i in range(rows):
#             for j in range(cols):
#                 os.remove(f"tile_{i}_{j}.png")

# from manim import *

# class Main(Scene):
#     def construct(self):
#         self.wait(5)
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
#         self.play(Create(ax),run_time=0.5)

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
#         self.play(FadeIn(points, lag_ratio=0.1), run_time=0.5)
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
#         self.play(FadeIn(regression_line), run_time=0.5)
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
#             run_time=1
#         )
#         for obj in self.mobjects:
#             obj.clear_updaters()
#         graph_group = Group(*self.mobjects)
#         self.play(graph_group.animate.scale(0.5).shift(UP*1.5+LEFT*3), run_time=0.5)
#         arrow1 = Arrow(LEFT,RIGHT, tip_length=0.2, stroke_width=3, color=YELLOW).scale([0.8,1,1]).next_to(graph_group, RIGHT)
#         arrow2 = Arrow(UP,DOWN, tip_length=0.2, stroke_width=3, color=YELLOW).scale([1,0.5,1]).next_to(graph_group, DOWN , aligned_edge=LEFT)
#         arrow3 = Arrow(UP,DOWN, tip_length=0.2, stroke_width=3, color=YELLOW).scale([1,0.5,1]).next_to(graph_group, DOWN , aligned_edge=RIGHT)
#         self.play(GrowArrow(arrow1),GrowArrow(arrow2),GrowArrow(arrow3))
#         logistic_regression = Text("Logistic Regression", font="Orbitron", color=TEAL).scale(0.3).next_to(arrow1, RIGHT)
#         neural_network = Text("Neural Networks", font="Orbitron", color=TEAL).scale(0.3).next_to(arrow2, DOWN)
#         svm = Text("Support Vector Machines", font="Orbitron", color=TEAL).scale(0.3).next_to(arrow3, DOWN)
#         self.play(FadeIn(logistic_regression,neural_network,svm))
#         self.wait(5)
#         self.play(FadeOut(arrow1,arrow2,arrow3,logistic_regression,neural_network,svm), graph_group.animate.shift(UP))



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
#         video_path = r"C:\Users\iT HOME\Desktop\Machine_Learning_For_Physicists\1_Linear_Regression\media\videos\main\2160p60\ParametricCylinderScene.mp4"  # Path to your video
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
#             img = ImageMobject(img_path).scale(0.1)
#             label = Text(f"t = {snapshot_times[i]}s", font="Orbitron", color=YELLOW).scale(0.3)
#             group = Group(img, label).arrange(DOWN*0.5)
#             snapshots.append(group)

#         # --- Step 3: Arrange all in a grid ---
#         grid = Group(*snapshots).arrange_in_grid(rows=4, cols=4, buff=(0,0)).scale(0.9)

#         for i, group in enumerate(snapshots):
#             self.play(FadeIn(group), run_time=0.3)
#             self.wait(0.1)
#         self.wait(2)

# from manim import *
# import numpy as np

# class LinearRegression1(Scene):
#     def construct(self):
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
#             def make_error_line(x_=x, y_=y):
#                 return always_redraw(lambda: DashedLine(
#                     start=ax.c2p(x_, y_),
#                     end=ax.c2p(x_, slope_tracker.get_value() * x_ + intercept_tracker.get_value()),
#                     color=PURE_GREEN,
#                     stroke_opacity=0.6
#                 ))
#             error_lines.add(make_error_line())
#         self.add(error_lines)

#         # Animate slope and intercept to "wobble" into place
#         self.play(
#             slope_tracker.animate.set_value(true_slope + 0.4),
#             intercept_tracker.animate.set_value(true_intercept - 0.5),
#             run_time=1
#         )
#         self.play(
#             slope_tracker.animate.set_value(true_slope - 0.2),
#             intercept_tracker.animate.set_value(true_intercept + 0.3),
#             run_time=0.7
#         )
#         self.play(
#             slope_tracker.animate.set_value(true_slope + 0.1),
#             intercept_tracker.animate.set_value(true_intercept - 0.1),
#             run_time=0.5
#         )
#         self.play(
#             slope_tracker.animate.set_value(true_slope),
#             intercept_tracker.animate.set_value(true_intercept),
#             run_time=0.5
#         )
#         self.wait()

# class LogisticRegressionScene2(Scene):

#     def construct(self):

#         # build the logistic regression model
#         url = r"https://raw.githubusercontent.com/thomasnield/machine-learning-demo-data/master/classification/simple_logistic_regression.csv"

#         data, m_tracker, b_tracker, ax, points, true_points, false_points, \
#             plot, f, max_line, likelihood_lines = create_model(data=pd.read_csv(url),
#                                                                initial_m=0.5,
#                                                                initial_b=-0.5
#                                                                )

#         x_label = ax.get_x_axis_label(
#             Text("Temperature (°C)").scale(0.4), edge=DOWN, direction=DOWN, buff=0.5
#         )
#         y_label = ax.get_y_axis_label(
#             Text("Ice Cream Sales").scale(0.4).rotate(90 * DEGREES),
#             edge=LEFT,
#             direction=LEFT,
#             buff=0.3,
#         )
#         self.play(LaggedStartMap(Write, ax),
#                   Write(max_line),
#                   Write(MathTex("0") \
#                         .scale(.8) \
#                         .next_to(ax.c2p(0, 0), DL, buff=.2)
#                         ),
#                   Write(x_label),Write(y_label)
#                   )

#         self.play(LaggedStartMap(Write, VGroup(*true_points)))
#         self.play(LaggedStartMap(Write, VGroup(*false_points)))
#         self.play(Write(plot))
#         self.play(LaggedStartMap(Write, VGroup(*likelihood_lines)))

#         self.play(
#             m_tracker.animate.set_value(0.5),
#             b_tracker.animate.set_value(-1.17576395),
#             run_time=1
#         )
#         self.play(
#             m_tracker.animate.set_value(0.33),
#             b_tracker.animate.set_value(-3.17576395),
#             run_time=1
#         )
#         self.play(
#             m_tracker.animate.set_value(0.19267212),
#             b_tracker.animate.set_value(-3.17576395),
#             run_time=1
#         )
#         self.play(
#             m_tracker.animate.set_value(0.9),
#             b_tracker.animate.set_value(-3.5),
#             run_time=1
#         )
        
#         self.wait()




# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# plt.style.use("dark_background")

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.grid(False)
# ax.xaxis.pane.set_visible(False)
# ax.yaxis.pane.set_visible(False)
# ax.zaxis.pane.set_visible(False)
# ax.set_facecolor("black")

# # Generate data
# np.random.seed(0)
# original_class1 = np.random.normal(loc=[-2, -2, 2], scale=0.5, size=(8, 3))
# original_class2 = np.random.normal(loc=[2, 2, -1], scale=0.5, size=(8, 3))
# support_vectors = np.array([original_class1[0], original_class2[0]])

# # Placeholders
# scatter1 = [ax.scatter(*original_class1.T, color='blue', s=50, label='Class +1')]
# scatter2 = [ax.scatter(*original_class2.T, color='red', s=50, label='Class -1')]
# support_scatter = [ax.scatter(*support_vectors.T, color='yellow', s=100,
#                               edgecolor='black', linewidth=1.5, label='Support Vectors')]

# # Axis settings
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)
# ax.set_zlim(-2, 3)
# ax.set_xlabel("X Position")
# ax.set_ylabel("Y Position")
# ax.set_zlabel("Z Position")
# ax.legend()

# # Plane mesh
# xx, yy = np.meshgrid(np.linspace(-2, 2, 2), np.linspace(-2, 2, 2))
# a, b, c, d = 1, 1, -1, 0
# zz = (-a * xx - b * yy - d) / c
# plane_points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
# plane_poly = [None]

# def rotate_points(points, angles):
#     ax_angle, ay_angle, az_angle = angles
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(ax_angle), -np.sin(ax_angle)],
#         [0, np.sin(ax_angle),  np.cos(ax_angle)]
#     ])
#     Ry = np.array([
#         [ np.cos(ay_angle), 0, np.sin(ay_angle)],
#         [0, 1, 0],
#         [-np.sin(ay_angle), 0, np.cos(ay_angle)]
#     ])
#     Rz = np.array([
#         [np.cos(az_angle), -np.sin(az_angle), 0],
#         [np.sin(az_angle),  np.cos(az_angle), 0],
#         [0, 0, 1]
#     ])
#     R = Rz @ Ry @ Rx
#     return points @ R.T

# def update(frame):
#     if plane_poly[0] is not None:
#         plane_poly[0].remove()
#     scatter1[0].remove()
#     scatter2[0].remove()
#     support_scatter[0].remove()

#     # --- Plane Rotation ---
#     ax_angle = np.radians(10 * np.sin(frame / 20))
#     ay_angle = np.radians(20 * np.sin(frame / 30))
#     az_angle = np.radians(15 * np.sin(frame / 40))
#     plane_angles = (ax_angle, ay_angle, az_angle)

#     # Rotate plane
#     rotated_plane = rotate_points(plane_points, plane_angles).reshape(2, 2, 3)
#     verts = [[rotated_plane[0,0], rotated_plane[0,1], rotated_plane[1,1], rotated_plane[1,0]]]
#     plane_poly[0] = ax.add_collection3d(Poly3DCollection(
#         verts, color='white', alpha=0.4, edgecolor='gray'))

#     # --- Points Respond to Plane Rotation Dynamically ---
#     # They get a mix of:
#     # - some of the plane’s rotation
#     # - a bit of local independent wiggling motion

#     # Use lower amplitude rotations for points (not fully following the plane)
#     point_angles = tuple(0.5 * a for a in plane_angles)

#     # Slight additional oscillation for variety
#     wiggle = lambda i, f: 0.05 * np.sin(0.1 * f + i)

#     def transform_points(points, frame):
#         pts = rotate_points(points, point_angles).copy()
#         for i in range(len(pts)):
#             pts[i] += np.array([
#                 wiggle(i, frame),
#                 wiggle(i + 10, frame),
#                 wiggle(i + 20, frame)
#             ])
#         return pts

#     new_class1 = transform_points(original_class1, frame)
#     new_class2 = transform_points(original_class2, frame)
#     new_support = transform_points(support_vectors, frame)

#     scatter1[0] = ax.scatter(*new_class1.T, color='blue', s=50)
#     scatter2[0] = ax.scatter(*new_class2.T, color='red', s=50)
#     support_scatter[0] = ax.scatter(*new_support.T, color='yellow', s=100,
#                                     edgecolor='black', linewidth=1.5)

#     # Optional camera movement
#     ax.view_init(elev=25, azim=30 + 0.5 * frame)

# # Animate
# ani = animation.FuncAnimation(fig, update, frames=100, interval=70)
# ani.save("svm_plane_dynamic_points.mp4", writer="ffmpeg", dpi=300, fps=20)
# plt.show()

# from manim import *
# import random
# class Segment2(Scene):
#     def construct(self):
#         self.wait(1.6)
#         box1 = SVGMobject("box.svg").scale(0.5).shift(UP*2+LEFT*2).set_opacity(0)
#         self.add(box1)
#         box2 = SVGMobject("box.svg").scale(0.5).set_opacity(0)
#         self.add(box2)
#         box3 = SVGMobject("box.svg").scale(0.5).shift(UP*2+RIGHT*2).set_opacity(0)
#         self.add(box3)
#         self.play(box1.animate.set_opacity(1).shift(LEFT*2+DOWN),box2.animate.set_opacity(1).shift(DOWN*3),box3.animate.set_opacity(1).shift(RIGHT*2+DOWN))
#         self.wait(0.5)
#         eq1 = MathTex(
#             r"\begin{aligned}"
#             r"&\text{Model:} && \hat{y} = \mathbf{w} \cdot \mathbf{x} + b \\"
#             r"&\text{Loss:} && \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2 \\"
#             r"&\text{Objective:} && \min_{\mathbf{w}, b} \ \mathcal{L}"
#             r"\end{aligned}",
#             color=YELLOW
#         ).scale(0.4).move_to(box1).shift(UP*1.3+LEFT)
#         eq2 = MathTex(
#             r"\begin{aligned}"
#             r"&\text{Decision Boundary:} && \mathbf{w} \cdot \mathbf{x} + b = 0 \\"
#             r"&\text{Constraint:} && y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \\"
#             r"&\text{Objective:} && \min_{\mathbf{w}, b} \ \frac{1}{2} \|\mathbf{w}\|^2"
#             r"\end{aligned}",
#             color=YELLOW
#         ).scale(0.4).move_to(box3).shift(UP*1.5+RIGHT*1.3)
#         eq3 = MathTex(
#             r"\begin{aligned}"
#             r"&\text{Model:} && \hat{y} = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}} \\"
#             r"&\text{Loss:} && \mathcal{L} = -\left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right] \\"
#             r"&\text{Objective:} && \min_{\mathbf{w}, b} \ \frac{1}{N} \sum_{i=1}^N \mathcal{L}^{(i)}"
#             r"\end{aligned}",
#             color=YELLOW
#         ).scale(0.4).move_to(box2).shift(UP*1.4)
#         self.play(Dot().set_opacity(0).move_to(box1).animate.become(eq1),
#                   Dot().set_opacity(0).move_to(box3).animate.become(eq2),
#                   Dot().set_opacity(0).move_to(box2).animate.become(eq3))
#         self.wait()
#         self.play(FadeOut(*self.mobjects))


#         center_dot = Dot(point=ORIGIN, radius=0.05, color=YELLOW).set_opacity(0)

#         # List of applications
#         applications = [
#             "House Price\nPrediction",
#             "Medical Risk\nScoring",
#             "Retail Demand\nForecasting",
#             "Student\nPerformance",
#             "Salary vs\nExperience",
#             "Simple Stock\nReturns"
#         ]

#         inner_radius = 1.2   # Arrow starting radius (to avoid overlapping image)
#         outer_radius = 3.2   # Where arrows end

#         arrows = VGroup()
#         labels = VGroup()
#         num = len(applications)

#         for i, app in enumerate(applications):
#             angle = i * 2 * PI / num

#             # Start and end points along a circle
#             start = inner_radius * np.array([np.cos(angle), np.sin(angle), 0])
#             end = outer_radius * np.array([np.cos(angle), np.sin(angle), 0])

#             arrow = Arrow(start=start, end=end, buff=0, color=TEAL, tip_length=0.2)
#             label = Text(app, font="Orbitron").set_fill(color=YELLOW, opacity=1).scale(0.3).move_to(end + 0.6 * normalize(end))

#             arrows.add(arrow)
#             labels.add(label)

#         self.wait(2)
#         self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.1))
#         self.play(LaggedStartMap(FadeIn, labels, lag_ratio=0.1))
#         self.wait(5)
#         self.play(FadeOut(*self.mobjects))
#         self.wait(2)
#         axes = Axes(
#             x_range=[0, 100, 10],
#             y_range=[20, 90, 10],
#             x_length=10,
#             y_length=5,
#             axis_config={"include_tip": False},
#             x_axis_config={"numbers_to_include": [0, 50, 100]},
#             y_axis_config={"numbers_to_include": [22, 50, 85]},
#         ).scale(0.4)

#         # Labels
#         x_label = axes.get_x_axis_label("Time (minutes)").scale(0.5).set_color(YELLOW).shift(DOWN*0.5)
#         y_label = axes.get_y_axis_label("Temperature (°C)").scale(0.5).set_color(YELLOW).shift(LEFT*0.5)
#         labels = VGroup(x_label, y_label)

#         # Room temperature line
#         room_temp = 25
#         x_min = axes.x_range[0]
#         x_max = axes.x_range[1]
#         room_line = DashedLine(
#             axes.c2p(x_min, room_temp),
#             axes.c2p(x_max, room_temp),
#             color=BLUE
#         )
#         room_label = Text("Room Temperature (Reservoir)").scale(0.3).next_to(room_line, LEFT).set_color(BLUE)

#         self.play(Create(room_line), FadeIn(room_label),Create(axes), Write(labels))

#         # Coffee cooling function: T(t) = T_room + (T0 - T_room) * exp(-kt)
#         T0 = 85
#         k = 0.03
#         cooling_func = lambda t: room_temp + (T0 - room_temp) * np.exp(-k * t)

#         # Dot and trace of coffee temperature
#         dot1 = Dot(color=RED)
#         dot1.move_to(axes.c2p(0, cooling_func(0)))
#         coffee_label = always_redraw(
#             lambda: Text("Coffee Temperature (System)").scale(0.3)
#             .next_to(dot1, DOWN*2)
#             .set_color(RED)
#         )
#         tracker = ValueTracker(0)
#         temp_label = always_redraw(
#             lambda: Text(f"{cooling_func(tracker.get_value()):.1f}°C", font_size=24)
#             .next_to(dot1, UP)
#             .set_color(RED)
#         )

#         # Graph line being drawn dynamically
#         graph = always_redraw(
#             lambda: axes.plot(
#                 cooling_func,
#                 x_range=[0, tracker.get_value()],
#                 color=RED
#             )
#         )

#         # Dot follows the graph
#         dot1.add_updater(lambda m: m.move_to(axes.c2p(tracker.get_value(), cooling_func(tracker.get_value()))))


#         self.play(FadeIn(dot1, temp_label, coffee_label), Create(graph))

#         # Animate cooling
#         self.play(tracker.animate.set_value(100), run_time=2)

#         self.wait()
#         graph = Group(dot1, temp_label, coffee_label,graph,room_line,room_label,axes,x_label,y_label)
#         self.play(graph.animate.shift(LEFT*2+DOWN*2))
#         reservoir = Square(side_length=5).scale(0.7).set_fill(YELLOW,opacity=0.2).set_stroke(color=WHITE).shift(RIGHT)

#         # System (small square inside reservoir)
#         system = Square(side_length=2).scale(0.5).set_fill(TEAL, opacity=0.2).set_stroke(color=WHITE)
#         system.move_to(reservoir.get_center())

#         # Red bidirectional arrow (energy exchange)
#         arrow = DoubleArrow(
#             start=system.get_left() + LEFT * 0.3,
#             end=system.get_left() + RIGHT * 0.3,
#             color=RED,
#             buff=0.1
#         )
       
#         h_sys = MathTex(r"\mathcal{H}(\Gamma_S)").scale(0.4)
#         h_sys.move_to(system)
#         system_label = Text("System", font="Arial", weight=BOLD).scale(0.3)
#         system_label.next_to(h_sys, DOWN, buff=0.2)

#         # Reservoir Hamiltonian label
#         h_res = MathTex(r"\mathcal{H}(\Gamma_R)").scale(0.7)
#         h_res.move_to(reservoir.get_corner(DL) + UR * 0.8)
#         res_label = Text("Reservoir", font="Arial", weight=BOLD).scale(0.4)
#         res_label.next_to(h_res, DOWN, buff=0.1)

#         self.play(Create(reservoir), Create(h_res), Create(res_label))
#         self.play(FadeIn(system,h_sys,system_label))
#         self.play(Create(arrow))
#         self.wait(6)
#         group2 = Group(reservoir, system, arrow, h_sys, system_label, h_res, res_label)
#         self.play(FadeOut(group2))
#         # ----- container settings -----
#         W, H = 3, 3                  # inner width & height in scene units
#         R    = 0.04                  # particle radius
#         BUFFER = 0.1                # distance before the wall to reverse

#         # Draw the translucent box
#         wall = (
#             Rectangle(width=W, height=H, stroke_width=2)
#             .set_fill(YELLOW, opacity=0.10)
#             .move_to(ORIGIN)
#         )
#         self.add(wall)

#         # ----- particle settings -----
#         N           = 25             # number of particles
#         SPEED_MIN   = 1.0
#         SPEED_MAX   = 2.5

#         particles = VGroup()

#         # Calculate safe spawn bounds (inside walls by R)
#         half_W = W / 2
#         half_H = H / 2
#         spawn_xmin, spawn_xmax = -half_W + R, half_W - R
#         spawn_ymin, spawn_ymax = -half_H + R, half_H - R

#         for _ in range(N):
#             dot = Dot(radius=R, color=YELLOW)

#             # random start position
#             x0 = random.uniform(spawn_xmin, spawn_xmax)
#             y0 = random.uniform(spawn_ymin, spawn_ymax)
#             dot.move_to([x0, y0, 0])

#             # random unit velocity scaled by random speed
#             v = np.random.randn(2)
#             v /= np.linalg.norm(v)
#             v *= random.uniform(SPEED_MIN, SPEED_MAX)
#             dot.v = np.array([v[0], v[1], 0.0])

#             particles.add(dot)

#         self.add(particles)

#         # Pre‑compute early‑bounce boundaries (include R and BUFFER)
#         xmin = -half_W + R + BUFFER
#         xmax =  half_W - R - BUFFER
#         ymin = -half_H + R + BUFFER
#         ymax =  half_H - R - BUFFER

#         # ----- updater -----
#         def bounce(dot: Dot, dt: float):
#             # advance particle
#             dot.move_to(dot.get_center() + dot.v * dt)
#             x, y, _ = dot.get_center()

#             # left / right walls
#             if (x <= xmin and dot.v[0] < 0) or (x >= xmax and dot.v[0] > 0):
#                 dot.v[0] *= -1
#                 x = np.clip(x, xmin, xmax)
#                 dot.move_to([x, y, 0])

#             # bottom / top walls
#             if (y <= ymin and dot.v[1] < 0) or (y >= ymax and dot.v[1] > 0):
#                 dot.v[1] *= -1
#                 y = np.clip(y, ymin, ymax)
#                 dot.move_to([x, y, 0])

#         particles.add_updater(lambda mob, dt: [bounce(d, dt) for d in mob])

#         # run the animation
#         self.wait(5)

#         particles.clear_updaters()

#         group = Group(wall,particles)
#         self.play(group.animate.shift(UP+LEFT*3))
#         num_rows = 10
#         col_labels = [
#             MathTex(r"\text{time}~(t)", color=YELLOW),
#             MathTex(r"\text{particle}_1", color=YELLOW),
#             MathTex(r"\text{particle}_2", color=YELLOW),
#             MathTex(r"\text{particle}_3", color=YELLOW),
#             MathTex(r"\cdots", color=YELLOW),
#             MathTex(r"\text{particle}_N", color=YELLOW)
#         ]

#         # --- Data (symbolic) ---
#         table_data = []
#         for t in range(num_rows):
#             row = [MathTex(rf"t = {t}", color=YELLOW)]
#             row.append(MathTex(r"\left(x_{1},\, y_{1}\right),\left(vx_{1},\, vy_{1}\right)", color=YELLOW))
#             row.append(MathTex(r"\left(x_{2},\, y_{2}\right),\left(vx_{2},\, vy_{2}\right)", color=YELLOW))
#             row.append(MathTex(r"\left(x_{3},\, y_{3}\right),\left(vx_{3},\, vy_{3}\right)", color=YELLOW))
#             row.append(MathTex(r"\cdots", color=YELLOW))
#             row.append(MathTex(r"\left(x_{N},\, y_{N}\right),\left(vx_{N},\, vy_{N}\right)", color=YELLOW))
#             table_data.append(row)

#         # --- Create Manim Table ---
#         table = Table(
#             [[cell for cell in row] for row in table_data],
#             col_labels=col_labels,
#             include_outer_lines=True,
#             element_to_mobject=lambda elem: elem,
#             h_buff=0.6,
#             v_buff=0.5,
#             line_config={"stroke_color": TEAL}
#         ).scale(0.35).next_to(group, RIGHT*2, aligned_edge=UP)
#         headers = table.get_col_labels()
#         self.play(FadeIn(headers))
#         self.wait()

#         rows = table.get_rows()[1:]
#         for row in rows:
#     # Simulate state change: advance particles for a short "simulation step"
#             for dot in particles:
#                 # manually move each particle forward slightly
#                 dot.move_to(dot.get_center() + dot.v * 0.01)

#             # Create a snapshot of current particle state
#             snapshot_particles = particles.copy().clear_updaters()
#             snapshot_wall = wall.copy()
#             snapshot_group = Group(snapshot_wall, snapshot_particles)

#             # Transform snapshot into table row
#             self.play(ReplacementTransform(snapshot_group, row), run_time=0.4)
#             self.wait(0.1)        
#         self.play(Create(table.vertical_lines), Create(table.horizontal_lines))
#         self.wait()
#         self.play(table.animate.scale(0.4).shift(UP*2+RIGHT*2))
#         self.wait()
#         graph.clear_updaters()
#         laws = MathTex(
#             r"\begin{array}{ll}"
#             r"\text{Microcanonical Ensemble:} &\quad \text{Probability law: } P_i = \frac{1}{\Omega} \\[0.4cm]"
#             r"\text{Canonical Ensemble:} &\quad \text{Probability law: } P_i = \frac{e^{-\beta E_i}}{Z} \\[0.4cm]"
#             r"\text{Grand Canonical Ensemble:} &\quad \text{Probability law: } P_i = \frac{e^{-\beta (E_i - \mu N_i)}}{\mathcal{Z}} \\[0.4cm]"
#             r"\text{Quantum Ensemble:} &\quad \text{Probability law: } f(E) = \frac{1}{e^{(E - \mu)/k_BT} \pm 1}"
#             r"\end{array}"
#         ).set_color(PURE_GREEN).scale(0.35).move_to(ORIGIN).shift(RIGHT*1.3)
        
#         self.play(Write(laws))
#         self.wait(4)
#         rect1 = SurroundingRectangle(graph[6])
#         self.play(Create(rect1))
#         self.wait(5)
#         self.play(Indicate(laws))
#         arrow3 = CurvedArrow(laws.get_top(), table.get_left(), radius=-2, tip_length=0.2, color=ORANGE)
#         self.play(Create(arrow3))
#         equations = MathTex(
#             r"\begin{array}{ll}"
#             r"\text{Linear Regression:} & \hat{y} = wx + b \\[0.4cm]"
#             r"\text{Logistic Regression:} & P(y=1|x) = \sigma(wx + b) \\[0.4cm]"
#             r"\text{Binary Cross-Entropy Loss:} & \mathcal{L} = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})] \\[0.4cm]"
#             r"\text{Gradient Descent:} & \theta \leftarrow \theta - \eta \frac{d\mathcal{L}}{d\theta}"
#             r"\end{array}"
#         ).scale(0.3).next_to(table, DOWN, aligned_edge=LEFT)
#         self.wait(7)
#         self.play(FadeIn(equations))
#         self.wait(6)

from manim import *
import random

class Segment3(Scene):
    def construct(self):
        self.wait(4.5)
        reservoir = Square(side_length=5).scale(0.7).set_fill(YELLOW,opacity=0.2).set_stroke(color=WHITE)
        system = Square(side_length=2).scale(0.5).set_fill(TEAL, opacity=0.2).set_stroke(color=WHITE)
        system.move_to(reservoir.get_center())
        h_sys = MathTex(r"\mathcal{A}").scale(0.4)
        h_sys.move_to(system)
        h_res = MathTex(r"\mathcal{A}^{'}").scale(0.7)
        h_res.move_to(reservoir.get_corner(DL) + UR * 0.8)
        self.play(Create(system))
        self.play(Write(h_sys))
        self.wait(2)
        self.play(Create(reservoir), Write(h_res))
        self.wait(2)
        total_energy = MathTex(r"E_{\text{total}} = E_{r} + E_{r}^{'} = E^{(0)} = \text{constant}", color=ORANGE).scale(0.5).next_to(reservoir, RIGHT*2+UP*2)
        group1 = Group(reservoir,system,h_sys,h_res)
        self.wait(2)
        self.play(ReplacementTransform(group1.copy(),total_energy[0][:6]))
        self.wait(2)
        self.play(Indicate(Group(system,h_sys)))
        e1 = MathTex(r"E_{r}").scale(0.4).next_to(h_sys,DOWN*0.7)
        e2 = MathTex(r"E_{r}^{'}", color=YELLOW).scale(0.5).next_to(h_res,DOWN)
        self.play(FadeIn(e1))
        self.wait(0.5)
        self.play(Indicate(Group(reservoir,h_res)))
        self.play(FadeIn(e2))
        self.wait()
        self.play(Indicate(total_energy[0][:6]))
        self.wait()
        self.play(ReplacementTransform(e1.copy(),total_energy[0][7:9]),
                  ReplacementTransform(e2.copy(),total_energy[0][10:13]))
        self.play(FadeIn(total_energy[0][9]), FadeIn(total_energy[0][6]))
        self.play(FadeIn(total_energy[0][13:]))
        self.wait()
        coffee_title = Text("System A",font="Orbitron", color=YELLOW).scale(0.3)
        coffee_value = Text("Accessible microstates ≈ 10⁸⁰").scale(0.3)
        coffee_group = VGroup(coffee_title, coffee_value).arrange(DOWN, buff=0.3)

        # Room label
        room_title = Text("Reservoir A′",font="Orbitron", color=YELLOW).scale(0.3)
        room_value = Text("Accessible microstates ≈ 10¹⁰²⁴").scale(0.3)
        room_group = VGroup(room_title, room_value).arrange(DOWN, buff=0.3)

        labels = VGroup(coffee_group, room_group).arrange(DOWN*3).next_to(total_energy,DOWN*2, aligned_edge=LEFT)
        self.play(FadeIn(labels, shift=UP), run_time=2)

        energy_comparison_label = MathTex(r"E_r \ll E^{(0)}", color=ORANGE).scale(0.7).next_to(reservoir, UP, aligned_edge=LEFT)
        self.play(Write(energy_comparison_label))
        self.wait(2)
        
        microstates = VGroup()
        cols, rows = 18, 18  # you can increase for more granularity
        w = reservoir.width
        h = reservoir.height
        square_w = w / cols
        square_h = h / rows

        for i in range(cols):
            for j in range(rows):
                s = Square(side_length=square_w * 0.9)  # spacing
                x = -w / 2 + square_w * (i + 0.5)
                y = -h / 2 + square_h * (j + 0.5)
                s.move_to(reservoir.get_center() + [x, y, 0])
                s.set_fill(YELLOW, opacity=0.2)
                s.set_stroke(width=1, color=WHITE)
                microstates.add(s)

        self.play(LaggedStartMap(FadeIn, microstates, lag_ratio=0.01), FadeOut(reservoir), FadeToColor(h_sys, ORANGE), FadeToColor(h_res, ORANGE), FadeToColor(e1,ORANGE), FadeToColor(e2,ORANGE), run_time=2)
        microstates1 = VGroup()
        cols, rows = 25, 25  # you can increase for more granularity
        w = reservoir.width + 0.2
        h = reservoir.height + 0.2
        square_w = w / cols
        square_h = h / rows

        for i in range(cols):
            for j in range(rows):
                s = Square(side_length=square_w * 0.9)  # spacing
                x = -w / 2 + square_w * (i + 0.5)
                y = -h / 2 + square_h * (j + 0.5)
                s.move_to(reservoir.get_center() + [x, y, 0])
                s.set_fill(YELLOW, opacity=0.2)
                s.set_stroke(width=1, color=WHITE)
                microstates1.add(s)
        self.play(microstates.animate.become(microstates1))
        self.wait(5)
        prob_eq = MathTex(r"P_r \propto \Omega'(E_r') = \Omega'(E^{(0)} - E_r)", color=ORANGE).scale(0.7).next_to(microstates1, LEFT*1.5, aligned_edge=UP)
        prob_eq[0][2:].set_opacity(0)
        self.play(ReplacementTransform(system.copy(),prob_eq))
        self.wait(4)
        self.play(prob_eq[0][2:3].animate.set_opacity(1))
        self.wait()
        self.play(prob_eq[0][3:10].animate.set_opacity(1))
        self.wait()
        self.play(prob_eq[0][10:11].animate.set_opacity(1))
        self.play(prob_eq[0][11:13].animate.set_opacity(1))
        self.play(
                LaggedStart(
                    ReplacementTransform(prob_eq[0][5:10].copy(), prob_eq[0][13:]),
                    prob_eq[0][13:].animate.set_opacity(1),
                    lag_ratio=0.25 
                )
                )
        expansion_box = RoundedRectangle(height=1,width=5).set_fill(TEAL , opacity=0.2).next_to(prob_eq, UP*2.5, aligned_edge=LEFT).shift(LEFT*0.3)
        replacement_part1 = MathTex(r"\ln", color=ORANGE).scale(0.7).move_to(prob_eq[0][2], aligned_edge=DOWN)
        replacement_part2 = MathTex(r"\ln", color=ORANGE).scale(0.7).move_to(prob_eq[0][11], aligned_edge=DOWN)
        self.wait()
        self.play(prob_eq[0][:3].animate.shift(LEFT*0.4), FadeIn(replacement_part1),prob_eq[0][11:].animate.shift(RIGHT*0.3), FadeIn(replacement_part2))
        expansion = MathTex(
            r"\ln \Omega'(E^{(0)} - E_r) \approx \ln \Omega'(E^{(0)}) - "
            r"\left( \frac{\partial \ln \Omega'}{\partial E'} \right)_{E' = E^{(0)}} E_r",
            color=YELLOW
        ).scale(0.4).move_to(expansion_box)
        self.play(Create(expansion_box))
        self.play(Write(expansion))
        eq = MathTex(
            r"\begin{aligned}"
            r"\ln \Omega'(E^{(0)}) \\"
            r"\quad\quad\quad\quad\quad - \left( \frac{\partial \ln \Omega'}{\partial E'} \right)_{E' = E^{(0)}} E_r"
            r"\end{aligned}",
            color=ORANGE
        ).scale(0.7).move_to(replacement_part2, aligned_edge=UP).shift(UP*0.1)
        self.play(ReplacementTransform(Group(expansion[0][14:].copy(),prob_eq[0][11:],replacement_part2),eq))
        self.wait()
        expansion_box2 = RoundedRectangle(height=1,width=9.5).set_fill(TEAL , opacity=0.2).move_to(expansion_box, aligned_edge=LEFT)
        self.play(expansion_box.animate.become(expansion_box2))
        expansion2 = MathTex(r"\beta' = \beta = \frac{1}{kT}", color=PURE_GREEN).scale(0.4).next_to(expansion, RIGHT*2)
        self.play(Write(expansion2))
        replacement_part3 = MathTex(r"\beta", color=ORANGE).scale(0.7).move_to(eq[0][11]).shift(RIGHT*2.3)
        subsection1 = expansion2[0][3].copy()
        self.wait(4)
        self.play(
                LaggedStart(
                    subsection1.animate.become(replacement_part3),
                    eq[0][11:-2].animate.set_opacity(0),
                    eq[0][10].animate.shift(RIGHT * 2.2),
                    lag_ratio=0.2  # Adjust timing between animations
                )
            )
        expansion3 = MathTex(r"\ln \Omega'(E^{(0)} - E_r) \approx \text{const}", color=TEAL).scale(0.4).next_to(expansion2, RIGHT*2)
        self.play(Write(expansion3))
        replacement_part4 = MathTex(r"\text{const}", color=ORANGE).scale(0.7).move_to(eq[0][0]).shift(RIGHT*0.2)
        self.play(ReplacementTransform(Group(expansion3[0][14:].copy(),eq[0][:10]),replacement_part4))
        self.play(Group(eq,subsection1).animate.shift(UP*0.76+RIGHT*0.4))
        replacement_part5  = MathTex(r"P_r \propto e^{-\beta E_r}", color=ORANGE).scale(0.7).move_to(prob_eq, aligned_edge=LEFT+UP)
        subsection2 = Group(eq,replacement_part1,replacement_part2,subsection1,replacement_part4,prob_eq)
        self.play(ReplacementTransform(subsection2,replacement_part5))
        rect2 = SurroundingRectangle(replacement_part5[0][3:])
        self.play(Create(rect2))
           # Create axes
        axes = Axes(
            x_range=[-0.5, 5.5, 1],
            y_range=[0, 1.2, 0.2],
            x_length=6,
            y_length=3,
            axis_config={"include_tip": False, "stroke_color": GRAY_B},
        )
        axes_labels = axes.get_axis_labels(
            x_label=MathTex("E"),
            y_label=MathTex("P_{r}")
        )

        # Boltzmann factor: P ∝ exp(-βE)
        beta = 1
        energies = [1, 2, 3, 4]
        probabilities = [np.exp(-beta * E) for E in energies]
        max_prob = max(probabilities)

        # Bars on axes
        bars = VGroup()
        bar_width = 0.4
        for i, (E, P) in enumerate(zip(energies, probabilities)):
            x = E
            bar_height = P
            bar = Rectangle(
                width=bar_width,
                height=bar_height,
                color=BLUE,
                fill_opacity=0.7,
                stroke_width=0
            ).scale([1,2,1])
            bar.move_to(axes.c2p(x, bar_height / 2.5))  # center of bar at correct height
            bars.add(bar)

        # Curve: e^{-βE}
        boltz_curve = axes.plot(lambda x: np.exp(-beta * x), x_range=[0, 4.5], color=YELLOW)

        # Group everything for manipulation
        all_group = VGroup(axes, axes_labels, bars, boltz_curve)
        all_group.next_to(replacement_part5, RIGHT*2, aligned_edge=UP).shift(LEFT*2+UP)
        all_group.scale(0.5)

        self.play(Create(axes), Write(axes_labels))
        self.play(LaggedStart(*[GrowFromEdge(bar, edge=DOWN) for bar in bars], lag_ratio=0.2))
        self.play(Create(boltz_curve))
        self.wait(1)
        self.play(Uncreate(rect2), FadeOut(all_group))
        partition_function  = MathTex(
            r"Z = e^{-\beta E_0} + e^{-\beta E_1}  + \cdots",
            color=ORANGE
        ).scale(0.7).next_to(replacement_part5, DOWN, aligned_edge=LEFT)
        # self.play(FadeIn(partition_function[0][:2]))
        # self.play(ReplacementTransform(microstates1[-1].copy(), partition_function[0][2:7]))
        # self.play(FadeIn(partition_function[0][7]))
        # self.play(ReplacementTransform(microstates1[-2].copy(), partition_function[0][8:13]))
        # self.play(FadeIn(partition_function[0][13]))
        # self.play(FadeIn(partition_function[0][14:]))
        # self.wait()
        partition_function1  = MathTex(
            r"Z = \sum_{r} e^{-\beta E_r}",
            color=ORANGE
        ).scale(0.7).next_to(replacement_part5, DOWN, aligned_edge=LEFT)
        # self.play(partition_function.animate.become(partition_function1))
        canonical_eq = MathTex(
            r"P_r = \frac{e^{-\beta E_r}}{Z}, \quad \text{where} \quad Z = \sum_r e^{-\beta E_r}",
            color=ORANGE
        ).scale(0.6).move_to(replacement_part5,aligned_edge=LEFT+UP)
        self.wait(4)
        self.play(ReplacementTransform(Group(replacement_part5),canonical_eq))
        self.wait(3)
        rect2 = SurroundingRectangle(canonical_eq[0][16])
        self.play(Create(rect2))
        self.wait(3)
        sum_of_probabilities  = MathTex(
            r"\sum_r P_r = 1",
            color=ORANGE
        ).scale(0.7).next_to(replacement_part5, DOWN, aligned_edge=LEFT)
        sum_of_probabilities[0][-2:].set_opacity(0)
        self.play(canonical_eq[0][16].copy().animate.become(sum_of_probabilities))
        self.wait(2)
        self.play(sum_of_probabilities[0][-2:].animate.set_opacity(1))
        self.wait()
        self.play(FadeOut(*self.mobjects))
        self.wait()
        problem1 = Text("Handwritten digit classification (e.g. MNIST )", font="Orbitron", color=YELLOW).scale(0.3).shift(LEFT*2+UP*3)
        self.play(Write(problem1))
        dataset1 = ImageMobject("problem.png").next_to(problem1,DOWN*2,aligned_edge=LEFT)
        self.play(FadeIn(dataset1))
        self.wait()
        problem2 = Text("Canonical Ensemble", font="Orbitron", color=YELLOW).scale(0.3).next_to(problem1,DOWN*10, aligned_edge=LEFT)
        self.play(Write(problem2))
        columns = [
            "Time Step", 
            "Microstate (ID)", 
            "Energy of System A", 
            "Energy of Reservoir A'", 
            "Total Energy", 
            "Probability"
        ]

        # All data rows must use strings
        data = [
            ["0", "s₁", "2.1", "7.9", "10.0", "0.11"],
            ["1", "s₂", "3.3", "6.7", "10.0", "0.07"],
            ["2", "s₃", "1.5", "8.5", "10.0", "0.14"],
            ["3", "s₄", "2.8", "7.2", "10.0", "0.09"],
            ["\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots", "\\vdots"]
        ]

        # Full table: headers + data
        table_data = [columns] + data

        # Table creation — everything as strings
        table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1, "stroke_color":TEAL},
            element_to_mobject=lambda txt: (
                MathTex(txt, font_size=35, color=YELLOW) if txt == "\\vdots" else Text(txt, font_size=24, color=YELLOW)
            )
        )

        table.scale(0.3)
        table.next_to(problem2, DOWN*2, aligned_edge=LEFT)

        self.play(Create(table))
        self.wait(22)

        x1 = MathTex("x_1", color=ORANGE).scale(0.6)
        x2 = MathTex("x_2", color=ORANGE).scale(0.6)
        x3 = MathTex("x_3", color=ORANGE).scale(0.6)
        x4 = MathTex("x_4", color=ORANGE).scale(0.6)
        x5 = MathTex("x_5", color=ORANGE).scale(0.6)
        xn = MathTex("x_n", color=ORANGE).scale(0.6)
 
        variables = VGroup(x1, x2, x3, x4, x5, xn).arrange(RIGHT, buff=0.6)
        for i in range(6):
            cell = table.get_cell((1, i + 1))  # row=1 is header
            variables[i].move_to(cell.get_center())
        for i in range(6):
            self.play(
                ReplacementTransform(table.get_entries((1, i + 1)), variables[i]),
                run_time=0.4
            )        
        self.wait()
        

        n_vars = 6
        mean_y_vals = mean_y_vals = [1, 3, 5 , 7 , 9 , 11]
        colors = [RED, GREEN, BLUE, TEAL , YELLOW , ORANGE]
        labels = [r"\bar{x}_1", r"\bar{x}_2", r"\bar{x}_3", r"\bar{x}_4", r"\bar{x}_5", r"\bar{x}_n"]

        self.wait(2)
        axes = Axes(
            x_range=[-1, 10, 1],
            y_range=[-1, 12, 1],
            x_length=10,
            y_length=5,
            axis_config={"include_ticks": False, "color": GREY}
        ).scale(0.3).next_to(problem1, RIGHT*13, aligned_edge=UP)

        # Group for bars, labels, and dots
        mean_bars = VGroup()
        mean_labels = VGroup()
        dot_groups = VGroup()

        n_dots = 10
        y_spacing = 0.8
        x_center = 5.0  # mean horizontal location

        for i, (mean_y, color, label) in enumerate(zip(mean_y_vals, colors, labels)):
            # Mean horizontal bar
            bar = Line(
                start=axes.c2p(x_center - 3.5, mean_y),
                end=axes.c2p(x_center + 3.5, mean_y),
                color=color,
                stroke_width=2
            )
            mean_bars.add(bar)

            # Label \bar{x}_i
            label_tex = MathTex(label, color=color).scale(0.5)
            label_tex.next_to(bar, LEFT*1.5, buff=0.3)
            mean_labels.add(label_tex)

            # Dots (y fixed, x oscillates)
            dots = VGroup()
            for j in range(n_dots):
                y = mean_y
                x = x_center + random.uniform(-0.3, 0.3)
                dot = Dot(point=axes.c2p(x, y), color=color, radius=0.07)
                dot.base_y = y
                dot.base_x = x_center
                dot.amp = random.uniform(0.3, 0.6)
                dot.phase = random.uniform(0, 2 * np.pi)
                dots.add(dot)
            dot_groups.add(dots)
        self.add(axes,mean_bars,mean_labels)
        
        for dots in dot_groups:
            self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.05), run_time=0.5)
        t_tracker = ValueTracker(0)

        def make_x_osc_updater(dot):
            return lambda m: m.move_to(
                axes.c2p(
                    dot.base_x + dot.amp * np.sin(2 * t_tracker.get_value() + dot.phase),
                    dot.base_y
                )
            )

        for dots in dot_groups:
            for dot in dots:
                dot.add_updater(make_x_osc_updater(dot))

        # Animate time over 6 seconds
        self.play(t_tracker.animate.increment_value(6), run_time=2, rate_func=linear)

        # reservoir = Square(side_length=5).scale(0.4).set_fill(YELLOW,opacity=0.2).set_stroke(color=WHITE).shift(UP)
        # system = Square(side_length=2).scale(0.15).set_fill(TEAL, opacity=0.2).set_stroke(color=WHITE)
        # system.move_to(reservoir.get_center())
        # h_sys = MathTex(r"\mathcal{A}").scale(0.4)
        # h_sys.move_to(system)
        # h_res = MathTex(r"\mathcal{A}^{'}").scale(0.7)
        # h_res.move_to(reservoir.get_corner(DL) + UR * 0.3)
        # self.play(Create(system),Write(h_sys), Create(reservoir), Write(h_res))
        # system_group = Group(reservoir,system,h_sys,h_res)
        # axes1 = Axes(
        #     x_range=[0, 100, 10],
        #     y_range=[20, 90, 10],
        #     x_length=10,
        #     y_length=5,
        #     axis_config={"include_tip": False},
        #     x_axis_config={"numbers_to_include": [0, 50, 100]},
        #     y_axis_config={"numbers_to_include": [22, 50, 85]},
        # ).scale(0.3).next_to(reservoir, RIGHT*2, aligned_edge=DOWN)

        # # Labels
        # x_label = axes1.get_x_axis_label("Time (minutes)").scale(0.5).set_color(YELLOW).shift(DOWN*0.7)
        # y_label = axes1.get_y_axis_label("Temperature (°C)").scale(0.5).set_color(YELLOW).shift(LEFT*1.5)
        # labels = VGroup(x_label, y_label)

        # # Room temperature line
        # room_temp = 25
        # x_min = axes.x_range[0]
        # x_max = axes.x_range[1]
        # room_line = DashedLine(
        #     axes1.c2p(x_min, room_temp),
        #     axes1.c2p(100, room_temp),
        #     color=BLUE
        # )
        # room_label = Text("A'").scale(0.3).next_to(room_line, LEFT).set_color(BLUE)

        # self.play(Create(room_line), FadeIn(room_label),Create(axes1), Write(labels))

        # # Coffee cooling function: T(t) = T_room + (T0 - T_room) * exp(-kt)
        # T0 = 85
        # k = 0.03
        # cooling_func = lambda t: room_temp + (T0 - room_temp) * np.exp(-k * t)

        # # Dot and trace of coffee temperature
        # dot1 = Dot(color=RED).scale(0.5)
        # dot1.move_to(axes1.c2p(0, cooling_func(0)))
        # coffee_label = always_redraw(
        #     lambda: Text("A").scale(0.3)
        #     .next_to(dot1, DOWN*2)
        #     .set_color(RED)
        # )
        # tracker = ValueTracker(0)
        # graph = always_redraw(
        #     lambda: axes1.plot(
        #         cooling_func,
        #         x_range=[0, tracker.get_value()],
        #         color=RED
        #     )
        # )

        # # Dot follows the graph
        # dot1.add_updater(lambda m: m.move_to(axes1.c2p(tracker.get_value(), cooling_func(tracker.get_value()))))


        # self.play(FadeIn(dot1, coffee_label), Create(graph))

        # # Animate cooling
        # self.play(tracker.animate.set_value(100), run_time=2)

        # self.wait()
        # graph = Group(dot1, coffee_label,graph,room_line,room_label,axes1,x_label,y_label)
        # self.wait()

        # self.play(Indicate(Group(reservoir,h_res)))

        # rect3 = SurroundingRectangle(table.get_rows()[1]).scale([1.1,1,1])
        # self.play(Create(rect3))
        # self.wait()
        # self.play(FadeOut(problem1,dataset1,problem2, graph, rect3), table.animate.shift(RIGHT*4), system_group.animate.shift(RIGHT*2))
        # equation1 = MathTex(
        #     r"P(\vec{x}) = \frac{1}{Z} e^{-F(\vec{x})}", color=ORANGE
        # ).scale(0.5).next_to(reservoir,LEFT*19, aligned_edge=UP).shift(UP*0.2)
        # self.play(Write(equation1))
        # rect4 = SurroundingRectangle(equation1[0][11:]).scale(0.9)
        # self.play(Create(rect4))
        # self.wait()
        # rect5 = SurroundingRectangle(equation1[0][8:9], color=TEAL).scale(0.9)
        # self.play(Create(rect5))
        # self.wait()
        # equation2 = MathTex(r"\vec{x} - \vec{\bar{x}} \ll 1", color=ORANGE)
        # equation2.scale(0.7)
        # equation2.to_edge(LEFT).shift(UP)
        # self.play(FadeIn(equation2, shift=RIGHT))
        # self.wait()
        # self.play(FadeOut(equation2, shift=LEFT), Uncreate(rect4), Uncreate(rect5))
        # expansion_box = RoundedRectangle(height=1,width=5).set_fill(TEAL , opacity=0.2).next_to(equation1, UP*1.5, aligned_edge=LEFT).shift(LEFT*1.2)
        # equation3 = MathTex(
        #     r"F(\vec{x}) \approx F(\vec{\bar{x}}) + \frac{1}{2} (\vec{x} - \vec{\bar{x}}) \cdot \vec{\Sigma}^{-1} \cdot (\vec{x} - \vec{\bar{x}})",
        #     color=YELLOW
        # ).scale(0.4).move_to(expansion_box).shift(LEFT*0.5)
        # self.play(Create(expansion_box))
        # self.play(Write(equation3))
        # self.wait()
        # rect6 = SurroundingRectangle(equation3[0][25:28])
        # self.play(Create(rect6))
        # self.play(Indicate(Group(x1, x2, x3, x4, x5, xn)))
        # self.wait()
        # expansion_box1 = RoundedRectangle(height=1,width=7).set_fill(TEAL , opacity=0.2).move_to(expansion_box, aligned_edge=LEFT+UP)
        # self.play(expansion_box.animate.become(expansion_box1), Uncreate(rect6))
        # equation4 = MathTex(
        #     r"(\Sigma^{-1})_{ij} \equiv \left. \frac{\partial^2 F}{\partial x_i \, \partial x_j} \right|_{\vec{\bar{x}}}",
        #     color=TEAL
        # ).scale(0.4).next_to(equation3, RIGHT*2, aligned_edge=UP)
        # self.play(Create(equation4))
        # rect7 = SurroundingRectangle(equation3[0][6:10], color=PURE_RED)
        # self.play(Create(rect7))
        # self.wait()
        # equation5 = MathTex(r"P(\vec{x}) = \frac{1}{(2\pi)^{n/2} \sqrt{\det \vec{\Sigma}}} \exp\left[ -\frac{1}{2} (\vec{x} - \vec{\bar{x}}) \cdot \vec{\Sigma}^{-1} \cdot (\vec{x} - \vec{\bar{x}}) \right]",
        #                     color=ORANGE).scale(0.5).move_to(equation1, aligned_edge=LEFT+UP)
        # self.play(equation1.animate.become(equation5), Uncreate(rect7))
        # self.wait()
        # headers = [
        #     MathTex("y", color=ORANGE),
        #     MathTex("X", color=ORANGE)
        # ]

        # # Step 2: Create 4 rows of sample data + 1 row of vertical dots
        # data = [
        #     ["1", "2.3"],
        #     ["0", "1.7"],
        #     ["1", "3.1"],
        #     ["0", "2.0"],
        #     [MathTex(r"\vdots", color=YELLOW), MathTex(r"\vdots",color=YELLOW)]
        # ]

        # # Step 3: Create the table
        # table2 = Table(
        #     [[*row] for row in data],  # Ensure each row is list
        #     col_labels=headers,
        #     include_outer_lines=True,
        #     line_config={"stroke_width": 1, "stroke_color":TEAL},
        #     element_to_mobject=lambda elem: elem if isinstance(elem, Mobject) else Text(str(elem), font_size=24, color=YELLOW)
        # ).scale(0.3).move_to(table,aligned_edge=LEFT+UP).shift(RIGHT*0.9)
        # self.play(table.animate.become(table2))
        # equation6 =   MathTex(
        #     r"""
        #     \begin{array}{l}
        #     \vec{X} = (x_1, x_2, \dots, x_{n-1}, y) \\
        #     \text{where} \quad X = (x_1, \dots, x_{n-1})
        #     \end{array}
        # """,
        # color=ORANGE
        # ).scale(0.7).next_to(table2, RIGHT*3, aligned_edge=UP)
        # self.play(FadeIn(equation6, shift=LEFT*3.5))
        # self.wait()
        # rect8 = SurroundingRectangle(table.get_entries((1,1)))
        # self.play(Create(rect8))
        # rect9 = SurroundingRectangle(table.get_entries((1,1))).shift(RIGHT*0.5)
        # self.play(Create(rect9))
        # self.wait()
        # equation7 = MathTex(r"P(y \mid X) = \frac{P(X, y)}{\int dy \, P(X, y)}",color=ORANGE).scale(0.5).next_to(equation5, DOWN*2, aligned_edge=LEFT)
        # self.play(FadeIn(equation7),Uncreate(rect8), Uncreate(rect9))
        # self.wait()
        # equation8 = MathTex(r"\hat{y} = \text{E}[y]= \int dy \, P(y \mid X) \, y",color=ORANGE).scale(0.5).next_to(equation7, DOWN*2, aligned_edge=LEFT)
        # self.play(FadeIn(equation8))
        # self.wait()
        # rect10 = SurroundingRectangle(equation5)
        # self.play(Create(rect10))
        # self.wait()
        # self.play(FadeOut(rect10))
        # equation9 = MathTex(r"\hat{y} = \beta_0 + \beta \cdot X",color=ORANGE).scale(0.5).next_to(equation8, DOWN*2, aligned_edge=LEFT)
        # group5 = Group(equation5.copy(), equation8.copy())
        # self.play(ReplacementTransform(group5,equation9))
        # self.wait()
        # expansion_box2 = RoundedRectangle(height=1,width=9.7).set_fill(TEAL , opacity=0.2).move_to(expansion_box, aligned_edge=LEFT+UP)
        # self.play(expansion_box.animate.become(expansion_box2))
        # equation10 = MathTex(r"\beta_0 = \bar{y}",color=PURE_GREEN).scale(0.4).next_to(equation4, RIGHT*2).shift(UP*0.3)
        # equation11 = MathTex(r"\beta_i = -\frac{1}{(\Sigma^{-1})_{nn}} \sum_{i=1}^{n-1} (\Sigma^{-1})_{in}(x_i - \bar{x}_i)",color=PURE_GREEN).scale(0.4).next_to(equation4, RIGHT*2).shift(DOWN*0.1)
        # self.play(equation9[0][3:5].copy().animate.become(equation10), equation9[0][7:].copy().animate.become(equation11))
        # self.play(FadeOut(equation1,equation7,equation8))
        # self.play(equation9.animate.shift(UP*2))
        # arrow1 = Arrow(start=equation9.get_top(),
        #                end=equation9.get_top() +  UP,
        #                stroke_width=2)
        # label1 = Text("Hooke's Law:\nF = -kx", font="Orbitron", color=YELLOW).scale(0.4).next_to(arrow1, UP)

        # # Arrow 2 (right side)
        # arrow2 = Arrow(start=equation9.get_right() ,
        #                end=equation9.get_right() + RIGHT,
        #                stroke_width=2)
        # label2 = Text("Ohm's Law:\nV = IR", font="Orbitron", color=YELLOW).scale(0.4).next_to(arrow2.get_end(), RIGHT)

        # # Arrow 3 (lower left)
        # arrow3 = Arrow(start=equation9.get_bottom() ,
        #                end=equation9.get_bottom()+ DOWN,
        #                stroke_width=2)
        # label3 = Text("Newton's 2nd Law:\nF = ma", font="Orbitron", color=YELLOW).scale(0.4).next_to(arrow3.get_end(), DOWN)
        # self.play(Create(arrow1), Write(label1),
        #           Create(arrow2), Write(label2),
        #           Create(arrow3), Write(label3))
        # self.wait()
        # rect11 = SurroundingRectangle(reservoir, color=PINK)
        # rect12 = SurroundingRectangle(Group(table, equation6),color=PINK)
        # self.play(Create(rect11))
        # self.play(Create(rect12))
        # self.wait()
        # self.play(FadeOut(*self.mobjects))
        # self.wait()
        # entropy = Text("Entropy", font="Orbitron", color=YELLOW).scale(0.6).shift(UP*2.8)
        # self.play(DrawBorderThenFill(entropy))
        # second_law = MathTex(
        #     r"\oint \frac{\delta Q}{T} \leq 0", color=ORANGE
        # ).scale(0.5).shift(LEFT*4)
        # max_entropy = MathTex(
        #     r"\max_{P} \left( -\sum_{i} P(x_i) \log P(x_i) \right)",
        #     color=ORANGE
        # ).scale(0.5).shift(RIGHT*4)
        # self.play(Create(second_law))
        # self.wait()
        # self.play(Create(max_entropy))


        # box = Square(side_length=2).shift(DOWN*1.5+LEFT )
        # self.add(box)

        # # Create particles with motion life
        # particles = self.create_particles(count=50, box=box)

        # self.add(*particles)

        # # Animate particles until they all stop
        # self.play(
        #     UpdateFromFunc(Group(*particles), lambda m: self.move_until_stop(m, box)),
        #     run_time=4,
        #     rate_func=linear
        # )
        # self.wait()
        # self.play(Indicate(second_law),Indicate(max_entropy))
        # formula = MathTex(r"S = - \sum_i p_i \ln p_i", color=ORANGE).scale(0.6).next_to(entropy,DOWN*2)
        # self.play(Create(formula))
        # self.play(FadeOut(box,Group(*particles)))
        # self.wait()
        # expansion_box = RoundedRectangle(height=1, width=5).set_fill(color=TEAL, opacity=0.2).shift(LEFT*0.2+UP*0.5)
        # self.play(Create(expansion_box))
        # eq1 = MathTex(r"p(x)", color=YELLOW).scale(0.5).next_to(formula, DOWN*3+LEFT*3)
        # eq2 = MathTex(
        #     r"\log p(x) \approx \log p(\mu) - \frac{1}{2}(x - \mu)^T H (x - \mu)",
        #     color=YELLOW
        # ).scale(0.5).next_to(eq1, DOWN*2, aligned_edge=LEFT)
        # eq3 = MathTex(
        #     r"p(x) \approx p(\mu) \exp\left( -\frac{1}{2}(x - \mu)^T H (x - \mu) \right)",
        #     color=YELLOW
        # ).scale(0.5).next_to(eq2, DOWN*2, aligned_edge=LEFT)
        # eq4 = MathTex(
        #     r"p(x) \approx \frac{1}{Z} \exp\left( -\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu) \right)",
        #     color=YELLOW
        # ).scale(0.5).next_to(eq3, DOWN*2, aligned_edge=LEFT)

        # # Display each equation in sequence
        # self.play(FadeIn(eq1))
        # expansion_box1 = RoundedRectangle(height=2, width=5).set_fill(color=TEAL, opacity=0.2).move_to(expansion_box,aligned_edge=UP+LEFT)
        # self.play(expansion_box.animate.become(expansion_box1))
        # self.play(FadeIn(eq2))
        # expansion_box2 = RoundedRectangle(height=3, width=5).set_fill(color=TEAL, opacity=0.2).move_to(expansion_box,aligned_edge=UP+LEFT)
        # self.play(expansion_box.animate.become(expansion_box2))
        # self.play(FadeIn(eq3))
        # expansion_box3 = RoundedRectangle(height=4, width=5).set_fill(color=TEAL, opacity=0.2).move_to(expansion_box,aligned_edge=UP+LEFT)
        # self.play(expansion_box.animate.become(expansion_box3))
        # self.play(FadeIn(eq4))
        # equation10 = MathTex(r"\beta_0 = \bar{y}",color=PURE_GREEN).scale(0.4).next_to(entropy, LEFT*20+DOWN*13, aligned_edge=UP)
        # equation11 = MathTex(r"\beta_i = -\frac{1}{(\Sigma^{-1})_{nn}} \sum_{i=1}^{n-1} (\Sigma^{-1})_{in}(x_i - \bar{x}_i)",color=PURE_GREEN).scale(0.4).next_to(equation10, DOWN, aligned_edge=LEFT)
        # self.play(Create(equation10),Create(equation11))


    def create_particles(self, count, box):
        particles = []
        for _ in range(count):
            x = random.uniform(box.get_left()[0] + 0.2, box.get_center()[0] - 0.5)
            y = random.uniform(box.get_bottom()[1] + 0.2, box.get_top()[1] - 0.2)
            dot = Dot(point=np.array([x, y, 0]), radius=0.03, color=YELLOW)
            dot.velocity = np.array([
                random.uniform(-0.08, 0.08),
                random.uniform(-0.08, 0.08),
                0
            ])
            dot.steps_remaining = random.randint(20, 40)  # short life
            particles.append(dot)
        return particles

    def move_until_stop(self, particles, box):
        # Box bounds
        x_min = box.get_left()[0] + 0.1
        x_max = box.get_right()[0] - 0.1
        y_min = box.get_bottom()[1] + 0.1
        y_max = box.get_top()[1] - 0.1

        for p in particles:
            if p.steps_remaining > 0:
                pos = p.get_center()
                new_pos = pos + p.velocity

                # Constrain to box
                if not (x_min < new_pos[0] < x_max):
                    p.velocity[0] *= -1
                if not (y_min < new_pos[1] < y_max):
                    p.velocity[1] *= -1

                p.move_to(pos + p.velocity)
                p.steps_remaining -= 1
            else:
                p.velocity = np.zeros(3)  # fully stop

        return particles


# from manim import *
# import random

# class DynamicBouncingParticlesScene(ThreeDScene):
#     def construct(self):
#         self.set_camera_orientation(phi=65 * DEGREES, theta=-65 * DEGREES)

#         # Create two dynamic wireframe boxes
#         box_size = 1
#         red_box = Cube(side_length=box_size, stroke_color=PURE_RED, stroke_width=2)\
#             .set_fill(opacity=0).set_shade_in_3d(False).move_to(LEFT * 0.5)
#         blue_box = Cube(side_length=box_size, stroke_color=BLUE, stroke_width=2)\
#             .set_fill(opacity=0).set_shade_in_3d(False).move_to(RIGHT * 0.5)

#         self.add(red_box, blue_box)

#         # Get dynamic bounds
#         red_bounds = self.get_box_bounds(red_box)
#         blue_bounds = self.get_box_bounds(blue_box)

#         # Create particles: red moves faster
#         red_particles = self.create_particles(color=PURE_RED, bounds=red_bounds, speed=4.0)
#         blue_particles = self.create_particles(color=BLUE, bounds=blue_bounds, speed=1.0)

#         self.add(*red_particles, *blue_particles)

#         # Animate bouncing
#         self.play(
#             UpdateFromFunc(Group(*red_particles), lambda m: self.update_particles(m, red_bounds)),
#             UpdateFromFunc(Group(*blue_particles), lambda m: self.update_particles(m, blue_bounds)),
#             run_time=2,
#             rate_func=linear,
#         )

#     def get_box_bounds(self, cube: Cube):
#         center = cube.get_center()
#         half = cube.side_length / 2
#         return {
#             "min": center - half,
#             "max": center + half
#         }

#     def create_particles(self, color, bounds, count=10, speed=1.0):
#         particles = []
#         for _ in range(count):
#             pos = np.array([
#                 random.uniform(bounds["min"][i] + 0.1, bounds["max"][i] - 0.1)
#                 for i in range(3)
#             ])
#             sphere = Sphere(radius=0.02)
#             sphere.set_fill(color=color, opacity=1)
#             sphere.set_stroke(color=color, width=0.5)
#             sphere.move_to(pos)
#             sphere.velocity = np.array([
#                 random.uniform(-0.05, 0.05) * speed for _ in range(3)
#             ])
#             particles.append(sphere)
#         return particles

#     def update_particles(self, particles, bounds):
#         for p in particles:
#             pos = p.get_center()
#             vel = p.velocity
#             new_pos = pos + vel
#             for i in range(3):
#                 if new_pos[i] <= bounds["min"][i] + 0.02 or new_pos[i] >= bounds["max"][i] - 0.02:
#                     vel[i] *= -1
#             p.velocity = vel
#             p.move_to(pos + vel)
#         return particles


# from manim import *

# class MaxEntropyPrinciple(Scene):
#     def construct(self):
#         # STEP 1: show initial microstates
#         step1_title = Text("Step 1: Start with many possible microstates", font="Orbitron", color=TEAL).scale(0.4).to_edge(UP)
#         self.play(Write(step1_title))

#         boxes1 = VGroup(*[Square(side_length=1) for _ in range(6)])
#         boxes1.arrange(RIGHT, buff=1.2).next_to(step1_title, DOWN * 1.5)
#         self.play(FadeIn(boxes1))

#         microstates =  [
#             [5, 1, 1, 1, 2],     # Unique
#             [5, 5, 1, 0, 1],     # Unique
#             [3, 2, 3, 2, 0],     # Group C
#             [2, 3, 2, 3, 2],     # Group C
#             [1, 5, 3, 2, 2],     # Group C
#             [2, 2, 2, 2, 2],     # Uniform
#         ]

#         all_dots1 = VGroup()
#         for box, state in zip(boxes1, microstates):
#             dots = self.create_config(box, state)
#             all_dots1.add(dots)

#         self.play(FadeIn(all_dots1, lag_ratio=0.1))
#         self.wait(2)

#         boxes_with_particles = VGroup()
#         for box, dots in zip(boxes1, all_dots1):
#             boxes_with_particles.add(VGroup(box, dots))

#         step2_title = Text("Step 2: Group microstates by macrostate", font="Orbitron", color=TEAL).scale(0.4)
#         step2_title.next_to(step1_title, DOWN * 7, aligned_edge=LEFT)
#         self.play(Write(step2_title))
#         grouping = Group(boxes_with_particles[0].copy(), boxes_with_particles[2].copy(), boxes_with_particles[-1].copy()).arrange(DOWN).next_to(boxes_with_particles[0], DOWN*4, aligned_edge=LEFT)
#         copy1 = boxes_with_particles[1].copy().next_to(boxes_with_particles[1],DOWN*4, aligned_edge=LEFT)
#         copy2 = boxes_with_particles[3].copy().next_to(boxes_with_particles[3],DOWN*4, aligned_edge=LEFT)
#         copy3 = boxes_with_particles[4].copy().next_to(boxes_with_particles[4],DOWN*4, aligned_edge=LEFT)
#         self.play(FadeIn(grouping))
#         self.play(FadeIn(copy1,copy2,copy3))
#         step3_title = Text("Step 3: The most probable macrostate has the most microstates", font="Orbitron", color=TEAL).scale(0.4)
#         step3_title.next_to(step2_title, DOWN * 7, aligned_edge=LEFT)
#         self.play(Write(step3_title))
#         grouping2 = Group(boxes_with_particles[0].copy(), boxes_with_particles[2].copy(), boxes_with_particles[-1].copy()).arrange(RIGHT).next_to(step3_title, DOWN*2)
#         self.play(ReplacementTransform(grouping.copy(),grouping2))
#         final_message = Text("Most likely state = Maximum Entropy", font="Orbitron", color=PURE_GREEN).scale(0.5)
#         final_message.next_to(step3_title, DOWN * 7.5, aligned_edge=LEFT)
#         self.play(Write(final_message))

#     # ✅ This must be indented at the same level as construct
#     def create_config(self, box, bin_counts):
#         group = VGroup()
#         center = box.get_center()
#         width = box.width
#         height = box.height
#         dx = width / len(bin_counts)

#         for i, count in enumerate(bin_counts):
#             x = center[0] - width/2 + dx * (i + 0.5)
#             for j in range(count):
#                 y = center[1] - height/2 + 0.18 * j + 0.1
#                 dot = Dot(point=[x, y, 0], radius=0.04, color=YELLOW)
#                 group.add(dot)

#         return group



# from manim import *

# class EntropyGraph(Scene):
#     def construct(self):
#         # Axes: Time vs Entropy
#         axes = Axes(
#             x_range=[0, 10, 1],
#             y_range=[0, 1.2, 0.2],
#             x_length=6,
#             y_length=3,
#             axis_config={"include_tip": True, "numbers_to_exclude": [0]},
#         )

#         # Axis labels (manually)
#         x_label = Text("Time", font_size=24)
#         y_label = Text("Entropy", font_size=24)

#         x_label.next_to(axes.x_axis, DOWN)
#         y_label.next_to(axes.y_axis, LEFT)

#         self.play(Create(axes), Write(x_label), Write(y_label))

#         # Plot entropy curve: S(t) = 1 - exp(-t/2)
#         entropy_curve = axes.plot(
#             lambda t: 1 - np.exp(-t / 2),
#             x_range=[0, 10],
#             color=PURE_GREEN
#         )
#         self.play(Create(entropy_curve), run_time=3)

#         # Optional hint about what entropy means
#         hint = Text(
#             "Entropy ≈ number of ways to arrange particles",
#             font="Orbitron",
#             color=YELLOW
#         ).scale(0.5)
#         hint.next_to(axes, DOWN, buff=0.6).align_to(axes, RIGHT)
#         self.play(Write(hint))

#         equilibrium_point = entropy_curve.point_from_proportion(1.0)
#         dot = Dot(equilibrium_point, color=YELLOW)
#         label = Text("Equilibrium\n(Max Entropy)", font_size=22).next_to(dot, UP, buff=0.2)

#         self.play(FadeIn(dot), Write(label))
#         self.wait()

