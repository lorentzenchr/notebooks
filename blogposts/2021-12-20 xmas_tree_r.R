# t sin(t) Xmas tree, see https://community.wolfram.com/groups/-/m/t/175891
library(rgl)

t <- seq(0, 100, by = 0.7)^0.6
x <- t * c(sin(t), sin(t + pi))
y <- t * c(cos(t), cos(t + pi))
z <- -2 * c(t, t)
color <- rep(c("darkgreen", "gold"), each = length(t))

open3d(windowRect = c(100, 100, 600, 600), zoom = 0.9)
bg3d("black")
spheres3d(x, y, z, radius = 0.3, color = color)

# On screen (skip if export)
play3d(spin3d(axis = c(0, 0, 1), rpm = 4))

# Export (requires 3rd party tool "ImageMagick" resp. magick-package)
# movie3d(spin3d(axis = c(0, 0, 1), rpm = 4), duration = 30, dir = getwd())
