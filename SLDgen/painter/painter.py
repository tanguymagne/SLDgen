from pathlib import Path

import numpy as np
import pydiffvg
import svgwrite
import torch

from .initialize import initialize_control_points


def safe_divide(numerator: torch.Tensor, denominator: float) -> torch.Tensor:
    """Safely divide, returning zeros if denominator is 0."""
    if denominator != 0:
        return numerator / denominator
    else:
        return torch.zeros_like(numerator)


def bSplineBasis(ts, j, n, knots):
    """Recursively compute B-spline basis function using Cox-de Boor algorithm."""
    if n == 0:
        return ((knots[j] <= ts) * (ts < knots[j + 1])).to(float)
    # Combine lower-degree basis functions with weighted interpolation
    w1 = safe_divide(ts - knots[j], knots[j + n] - knots[j])
    w2 = safe_divide(knots[j + n + 1] - ts, knots[j + n + 1] - knots[j + 1])
    return w1 * bSplineBasis(ts, j, n - 1, knots) + w2 * bSplineBasis(ts, j + 1, n - 1, knots)


def getAllBSplineBasis(n_control_points, n_sample, n=3):
    """Generate all B-spline basis functions for the given control points."""
    # Set up knot vector and sample points
    n_knots = n_control_points + n + 1
    knots = torch.linspace(0, 1, n_knots, dtype=torch.float32)

    start_t = torch.linspace(0, 1, n_knots)[n]
    end_t = torch.linspace(0, 1, n_knots)[-n - 1]
    ts = torch.linspace(start_t, end_t, n_sample)

    # Compute all basis functions and stack them
    b_splines_bases = []
    for j in range(len(knots) - n - 1):
        b_splines_bases.append(bSplineBasis(ts, j, n, knots))
    return torch.stack(b_splines_bases, dim=0).to(torch.float32)


class SLDBSplinePainter(torch.nn.Module):
    def __init__(self, args, device=None, mask=None):
        super(SLDBSplinePainter, self).__init__()

        self.args = args
        self.mask = mask
        self.device = device
        self.canvas_width, self.canvas_height = args.render_size, args.render_size

    def get_polyline_2d(self):
        """Compute weighted B-spline basis and sample 2D curve."""
        # Get control points and weights, pruning inactive ones if enabled
        control_points = self.control_points
        weights = self.weights
        width = self.width

        if self.args.prune_low_weights:
            is_active = self.is_active_cp
            control_points = control_points[is_active]
            weights = weights[is_active]
            width = width[is_active]

        if self.args.fixed_endpoints:
            control_points = torch.cat([self.first2points, control_points, self.last2points], dim=0)
            weights = torch.cat([self.first2weights, weights, self.last2weights], dim=0)
            width = torch.cat([self.first2widths, width, self.last2widths], dim=0)

        # Compute the basis spline for the control points
        basis_spline = getAllBSplineBasis(len(control_points), n_sample=self.args.sampling_rate)
        basis_spline = basis_spline.to(self.device)

        # Apply weights and normalize basis spline
        basis_spline = basis_spline * weights.unsqueeze(1)
        sum_basis_spline = basis_spline.sum(axis=0)
        normalized_basis_spline = torch.where(
            sum_basis_spline == 0, torch.zeros_like(basis_spline), basis_spline / sum_basis_spline
        )
        self.normalized_basis_spline = normalized_basis_spline

        # Sample the curve and width using the weighted normalized basis spline and control points
        sampled_curve = (normalized_basis_spline.unsqueeze(-1) * control_points.unsqueeze(1)).sum(
            axis=0
        )
        self.sampled_curve2d = sampled_curve
        self.sampled_width = (normalized_basis_spline * width.unsqueeze(1)).sum(axis=0)

    def get_polyline(self):
        """Convert 2D curve to 3D by adding z-axis."""
        self.get_polyline_2d()
        # Add zeros for the z axis
        self.sampled_curve3d = torch.cat(
            [self.sampled_curve2d, torch.zeros_like(self.sampled_curve2d[:, :1])], dim=1
        )

    def set_shapes(self):
        """Create path shape from the sampled curve."""
        self.get_polyline()
        n_control_points_per_seg = torch.tensor(
            [0] * (len(self.sampled_curve2d) - 2), dtype=torch.int32
        ).contiguous()
        stroke_width = (
            self.sampled_width if self.args.width == "optim" else torch.tensor(self.args.width)
        )
        path = pydiffvg.Path(
            num_control_points=n_control_points_per_seg,
            points=self.sampled_curve2d,
            stroke_width=stroke_width,
            is_closed=False,
        )
        self.shapes = [path]

    def render_warp(self):
        """Render scene using pydiffvg."""
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups
        )
        img = _render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
        return img

    def get_image(self):
        """Render shapes and apply alpha blending."""
        self.set_shapes()
        # Render and blend with white background
        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=self.device
        ) * (1 - img[:, :, 3:4])
        # Convert to NCHW format for neural network processing
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # HWC -> NCHW
        return img

    def init_image(self):
        """Initialize control points, weights, and render first image."""
        # Initialize and scale control points to canvas
        control_points = initialize_control_points(args=self.args, mask=self.mask)
        control_points[:, 0] *= self.canvas_width
        control_points[:, 1] *= self.canvas_height

        if self.args.fixed_endpoints:
            self.first2points = (
                torch.tensor(
                    np.array(
                        [
                            control_points[0],
                            control_points[0],
                            control_points[0],
                            control_points[1],
                            control_points[1],
                        ]
                    ),
                    dtype=torch.float32,
                )
                .contiguous()
                .to(self.device)
            )
            self.first2weights = torch.ones(len(self.first2points), dtype=torch.float32).to(
                self.device
            )
            self.first2widths = torch.ones(len(self.first2points), dtype=torch.float32).to(
                self.device
            ) * (self.args.width if self.args.width != "optim" else 1.0)
            self.last2points = (
                torch.tensor(
                    np.array(
                        [
                            control_points[-2],
                            control_points[-2],
                            control_points[-1],
                            control_points[-1],
                            control_points[-1],
                        ]
                    ),
                    dtype=torch.float32,
                )
                .contiguous()
                .to(self.device)
            )
            self.last2weights = torch.ones(len(self.last2points), dtype=torch.float32).to(
                self.device
            )
            self.last2widths = torch.ones(len(self.last2points), dtype=torch.float32).to(
                self.device
            ) * (self.args.width if self.args.width != "optim" else 1.0)
            control_points = (
                torch.tensor(control_points[2:-2], dtype=torch.float32).contiguous().to(self.device)
            )
        else:
            control_points = (
                torch.tensor(control_points, dtype=torch.float32).contiguous().to(self.device)
            )

        # Initialize weights and active control point tracking
        self.control_points = control_points
        self.weights = torch.ones(len(control_points), dtype=torch.float32).to(self.device)
        self.is_active_cp = torch.ones(len(control_points), dtype=torch.bool).to(self.device)

        # Initiliaze stroke width
        if isinstance(self.args.width, float):
            self.width = (torch.ones(len(control_points)) * self.args.width).to(self.device)
        elif self.args.width == "optim":
            self.width = (
                torch.ones(len(control_points), dtype=torch.float32).contiguous().to(self.device)
            )

        # Create shape group with black stroke
        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]), fill_color=None, stroke_color=stroke_color
        )
        self.shape_groups = [path_group]

        return self.get_image()

    def post_process_params(self):
        """Clamp weights and width and deactivate low-weight control points."""
        self.weights.clamp_(min=0.001, max=1.0)
        self.is_active_cp = torch.logical_and(self.weights > 0.002, self.is_active_cp)

        self.width.clamp_(min=0.1, max=3.0)

    def parameters(self):
        """Set up gradient-enabled parameters for optimization."""
        # Enable gradients for control points
        self.control_points.requires_grad = True
        self.param_infos = []
        self.param_infos.append(
            {"params": [self.control_points], "name": "control_points", "lr_ratio": 1.0}
        )
        # Optionally enable gradients for weights with normalized learning rate
        if self.args.optimize_cp_weights:
            self.weights.requires_grad = True
            self.param_infos.append(
                {
                    "params": [self.weights],
                    "name": "weights",
                    "lr_ratio": 1.0 / max(self.canvas_width, self.canvas_height),
                }
            )

        if self.args.width == "optim":
            self.width.requires_grad = True
            self.param_infos.append(
                {
                    "params": [self.width],
                    "name": "widths",
                    "lr_ratio": 3.0 / max(self.canvas_width, self.canvas_height),
                }
            )

        return self.param_infos

    def save_svg(self, output_dir, name):
        """Save current path as SVG file."""
        self.set_shapes()

        if self.args.width != "optim":
            pydiffvg.save_svg(
                "{}/{}.svg".format(output_dir, name),
                self.canvas_width,
                self.canvas_height,
                self.shapes,
                self.shape_groups,
            )
        else:
            dwg = svgwrite.Drawing(
                "{}/{}.svg".format(output_dir, name),
                profile="tiny",
                size=(self.canvas_width, self.canvas_height),
            )

            for shape in self.shapes:
                points = shape.points.detach().cpu().numpy()
                stroke_width = shape.stroke_width.detach().cpu().numpy()

                tangent = points[1:] - points[:-1]
                average_tangent = tangent[:-1] + tangent[1:]

                all_points_tangents = np.concatenate(
                    [tangent[0:1], average_tangent, tangent[-1:]], axis=0
                )

                normals = np.zeros_like(all_points_tangents)
                normals[:, 0] = -all_points_tangents[:, 1]
                normals[:, 1] = all_points_tangents[:, 0]

                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / (norms + 1e-8)

                for i in range(len(points) - 1):
                    p0 = points[i]
                    p1 = points[i + 1]

                    n0 = normals[i]
                    n1 = normals[i + 1]

                    w0 = stroke_width[i]  # / 2.0
                    w1 = stroke_width[i + 1]  # / 2.0

                    path = dwg.path()
                    path.push("M", p0[0] + n0[0] * w0, p0[1] + n0[1] * w0)
                    path.push("L", p1[0] + n1[0] * w1, p1[1] + n1[1] * w1)
                    path.push("L", p1[0] - n1[0] * w1, p1[1] - n1[1] * w1)
                    path.push("L", p0[0] - n0[0] * w0, p0[1] - n0[1] * w0)
                    path.push("Z")  # Close the path

                    path.update(
                        {"stroke": "none", "fill": "black", "fill-opacity": 1.0, "stroke-width": 0}
                    )
                    dwg.add(path)

                dwg.save()

    def save_basis_spline(self, output_path):
        """Visualize and save basis spline functions and control point data."""
        # Save control point metadata to CSV
        with open(Path(output_path).with_suffix(".csv"), "w") as f:
            f.write("Active, Weights, CoordX, CoordY\n")
            for active, w, cp in zip(
                self.is_active_cp.detach().cpu().numpy(),
                self.weights.detach().cpu().numpy(),
                self.control_points.detach().cpu().numpy(),
            ):
                f.write(
                    f"{int(active) if self.args.prune_low_weights else 1}, {w}, {cp[0]}, {cp[1]}\n"
                )
