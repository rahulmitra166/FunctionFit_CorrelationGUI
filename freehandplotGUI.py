import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
from itertools import chain
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define function types
FUNCTION_TYPES = [
    'sine',
    'cosine',
    'exponential',
    'linear',
    'polynomial 2',
    'polynomial 3',
    'polynomial 4',
    'polynomial 5',
    'polynomial 6',
    'tangent',
    'hyperbolic tangent',
    'hyperbolic sine',
    'hyperbolic cosine',
    'logarithmic'
]

FUNCTION_NAMES = {
    'linear': 'y = a*x + b',
    'exponential': 'y = a*exp(b*x)',
    'logarithmic': 'y = a*ln(x) + b',
    'cosine': 'y = a*cos(b*x) + c',
    'sine': 'y = a*sin(b*x) + c',
    'tangent': 'y = a*tan(b*x) + c',
    'hyperbolic sine': 'y = a*sinh(b*x) + c',
    'hyperbolic cosine': 'y = a*cosh(b*x) + c',
    'hyperbolic tangent': 'y = a*tanh(b*x) + c',
    'polynomial 2': 'y = a*x^2 + b*x + c',
    'polynomial 3': 'y = a*x^3 + b*x^2 + c*x + d',
    'polynomial 4': 'y = a*x^4 + b*x^3 + c*x^2 + d*x + e',
    'polynomial 5': 'y = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f',
    'polynomial 6': 'y = a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + g',
}

FUNCTION_PARAMS = {
    'linear': ['slope', 'intercept'],
    'exponential': ['a', 'b'],
    'logarithmic': ['a', 'b'],
    'cosine': ['a', 'b', 'c', 'd'],
    'sine': ['a', 'b', 'c', 'd'],
    'tangent': ['a', 'b', 'c', 'd'],
    'hyperbolic sine': ['a', 'b', 'c', 'd'],
    'hyperbolic cosine': ['a', 'b', 'c', 'd'],
    'hyperbolic tangent': ['a', 'b', 'c', 'd'],
    'polynomial 2': ['a', 'b', 'c'],
    'polynomial 3': ['a', 'b', 'c', 'd'],
    'polynomial 4': ['a', 'b', 'c', 'd', 'e'],
    'polynomial 5': ['a', 'b', 'c', 'd', 'e', 'f'],
    'polynomial 6': ['a', 'b', 'c', 'd', 'e', 'f', 'g']
}

FUNCTION_MAP = {
    'exponential': (lambda x, a, b, c: a * np.exp(b * x) + c, [1.0, 0.0, 0.0]),
    'logarithmic': (lambda x, a, b: a * np.log(b * x), [1.0, 1.0]),
    'cosine': (lambda x, a, b, c, d: a * np.cos(b * x + c) + d, [1.0, 1.0, 0.0, 0.0]),
    'sine': (lambda x, a, b, c, d: a * np.sin(b * x + c) + d, [1.0, 1.0, 0.0, 0.0]),
    'tangent': (lambda x, a, b, c, d: a * np.tan(b * x + c) + d, [1.0, 1.0, 0.0, 0.0]),
    'hyperbolic sine': (lambda x, a, b, c: a * np.sinh(b * x) + c, [1.0, 1.0, 0.0]),
    'hyperbolic cosine': (lambda x, a, b, c: a * np.cosh(b * x) + c, [1.0, 1.0, 0.0]),
    'hyperbolic tangent': (lambda x, a, b, c: a * np.tanh(b * x) + c, [1.0, 1.0, 0.0]),
    'linear': (lambda x, a, b: a * x + b, [1.0, 0.0]),
    'polynomial 2': (lambda x, a, b, c: a * x**2 + b * x + c, [1.0, 1.0, 0.0]),
    'polynomial 3': (lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, [1.0, 1.0, 1.0, 0.0]),
    'polynomial 4': (lambda x, a, b, c, d, e: a * x**4 + b * x**3 + c * x**2 + d * x + e, [1.0, 1.0, 1.0, 1.0, 0.0]),
    'polynomial 5': (lambda x, a, b, c, d, e, f: a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f, 
                     [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    'polynomial 6': (lambda x, a, b, c, d, e, f, g: a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g, 
                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
} 

def calculate_correlation(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0
    corr, _ = pearsonr(x, y)
    return corr

class freehandplot:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(self.master, width=500, height=500)
        self.canvas.pack()

        # Draw axes and labels
        x_axis = self.canvas.create_line(50, 450, 450, 450, width=2)
        y_axis = self.canvas.create_line(50, 50, 50, 450, width=2)

        self.canvas_height = 500
        # Bind mouse events and set cursor
        self.canvas.bind('<B1-Motion>', self.on_motion)
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        self.canvas.configure(cursor='crosshair')

        # Initialize coordinates and line ID
        self.line_id = None
        self.coords = []
        self.fit_line_id = None

        # Create label for displaying correlation
        self.label = tk.Label(self.master, text='Correlation: 0.00')
        self.label.pack()

        # Create dropdown menu for selecting function type
        self.selected_func_type = tk.StringVar()
        self.selected_func_type.set(FUNCTION_TYPES[0])
        self.func_type_menu = tk.OptionMenu(self.master, self.selected_func_type, *FUNCTION_TYPES)
        self.func_type_menu.pack()

        # Create button for calculating correlation
        self.correlation_button = tk.Button(self.master, text='Calculate Correlation',
                                             command=self.update_correlation, state=tk.DISABLED)
        self.correlation_button.pack()

        # Create button for fitting function
        self.fit_button = tk.Button(self.master, text='Fit Function', command=self.fit_function, state=tk.DISABLED)
        self.fit_button.pack()

        # Create label for displaying best-fit function
        self.label_func = tk.Label(self.master, text='')
        self.label_func.pack()

        # Create button for refreshing canvas
        self.refresh_button = tk.Button(self.master, text='Refresh', command=self.refresh_canvas)
        self.refresh_button.pack()

        # Initialize min and max values for x and y
        self.min_x = 1
        self.max_x = 20
        self.min_y = 1
        self.max_y = 20

        # Initialize scale factor for converting between data coordinates and pixel coordinates
        self.scale_factor = 20

    def convert_coords_to_pixels(self, x, y):
        x_pixels = [(xi - 1) * self.scale_factor + 50 for xi in x]
        y_pixels = [450 - (yi - self.min_y) * self.scale_factor for yi in y]
        return list(chain(*zip(x_pixels, y_pixels)))

    def on_motion(self, event):
        if self.line_id is not None:
            x, y = event.x, event.y
            self.canvas.create_line(self.prev_x, self.prev_y, x, y, width=2, tags='line')
            self.coords.append((x - 250, 250 - y))
            self.prev_x, self.prev_y = x, y

    def on_button_press(self, event):
        if self.line_id is None:
            x, y = event.x, event.y
            self.line_id = self.canvas.create_line(x, y, x, y, width=2, tags='line')
            self.coords.append((x - 250, 250 - y))
            self.prev_x, self.prev_y = x, y

    def on_button_release(self, event):
        if self.line_id is not None:
            self.line_id = None
            self.correlation_button.config(state=tk.NORMAL)
            self.fit_button.config(state=tk.NORMAL)

    def update_correlation(self):
        # Get coordinates of the line
        x, y = zip(*self.coords)
        x = np.array(x)
        y = np.array(y)
        # Calculate correlation and update label
        corr = calculate_correlation(x, y)
        self.label.config(text='Correlation: {:.4f}'.format(corr))
        self.correlation_button.config(state=tk.DISABLED)
    
    def get_coordinates(self):
        x_coords = []
        y_coords = []
        for coord in self.coords:
            x_coords.append((coord[0] - 50) / self.scale_factor + 1)
            y_coords.append((450 - coord[1]) / self.scale_factor + 1)
        return x_coords, y_coords

    def fit_function(self):
        x, y = self.get_coordinates()
        x = np.array(x)
        y = np.array(y)

        # Convert pixel-based coordinates to actual data values
        x = np.array([(xi - self.max_x) / self.scale_factor + 1 for xi in x])
        y = np.array([(self.max_y - yi) / self.scale_factor + 1 for yi in y])

        # Get selected function type
        selected_func_type = self.selected_func_type.get()

        # Get function object and initial parameter guesses
        func, init_guess = FUNCTION_MAP[selected_func_type]

        # Fit the function to the data
        try:
            popt, _ = curve_fit(func, x, y, p0=init_guess)
        except RuntimeError:
            messagebox.showerror('Error', 'Failed to fit function to data.')
            return

        # Calculate correlation and update label
        score = calculate_correlation(func(x, *popt), y)
        #self.label.configure(text='Correlation: {:.2f}'.format(score))

        # Get best-fit function string and update label
        fit_func_str = self.get_fit_func_str(selected_func_type, popt)
        self.label_func.configure(text=fit_func_str)

        # Remove old best-fit function line and plot new one
        if self.fit_line_id is not None:
            self.canvas.delete(self.fit_line_id)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = func(x_fit, *popt)
        
        # Scatter plot of data points
        plt.scatter(x, y)

        # Plot best-fit line
        plt.plot(x_fit, y_fit, color='red', label='Best-fit line')

        # Add labels and legend
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        # Show the plot
        plt.show()

    def get_fit_func_str(self, selected_func_type, popt):
        func_name = FUNCTION_NAMES[selected_func_type]
        param_names = FUNCTION_PARAMS[selected_func_type]
        param_str = ', '.join(['{}={:.2f}'.format(name, val) for name, val in zip(param_names, popt)])
        return '{}({})'.format(func_name, param_str)

    def refresh_canvas(self):
        # Clear canvas
        self.canvas.delete('all')

        # Redraw axes and labels
        x_axis = self.canvas.create_line(50, 450, 450, 450, width=2)
        y_axis = self.canvas.create_line(50, 50, 50, 450, width=2)

        # Bind mouse events and set cursor
        self.canvas.bind('<B1-Motion>', self.on_motion)
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        self.canvas.configure(cursor='crosshair')

        # Reset line ID and coordinates
        self.line_id = None
        self.coords = []

        # Reset label and buttons
        self.label.config(text='Correlation: 0.00')
        self.correlation_button.config(state=tk.DISABLED)
        self.fit_button.config(state=tk.DISABLED)
        self.label_func.config(text='')

    def run(self):
        self.master.mainloop()


