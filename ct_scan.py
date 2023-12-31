# preface: this is only a demonstration of the basic math
# of computed tomography scan and reverse computed tomography

# the reverse part would not actually work
# due to rays not cancelling, not having the proper cancellation function
# frequencies requiring time dimension, high frequencies require high resolution

# time dimension is handled in fft_reverse_tomography.py  

# all these function are creating a single large matrix
# the work happens at the end

import torch
import math

def is_left(point0, point1, point):
    return (point1[0] - point0[0]) * (point[1] - point0[1]) - (point[0] - point0[0]) * (point1[1] - point0[1])

def intersect(pointA0, pointA1, pointB0, pointB1):
    u = (pointA1[0] - pointA0[0], pointA1[1] - pointA0[1])
    v = (pointB1[0] - pointB0[0], pointB1[1] - pointB0[1])
    denom = u[0] * v[1] - u[1] * v[0]
    w = is_left(pointB0, pointB1, pointA0) / denom
    pointC = (pointA0[0] + w * u[0], pointA0[1] + w * u[1])
    return pointC
    
def polygon_intersect_convexpolygon(polyA, polyB):
    #Sutherland-Hodgman
    inputList = polyA.copy()
    inputListSize = len(polyA)
    outputList = []
    
    j = len(polyB) - 1
    for k in range(len(polyB)):
        
        if inputListSize:
            prev_point = inputList[inputListSize - 1];
        for i in range(inputListSize):  
            current_point = inputList[i];

            currentpoint_inside_clipedge = (is_left(polyB[j], polyB[k], current_point) > 0)
            prevpoint_inside_clipedge =    (is_left(polyB[j], polyB[k], prev_point) > 0) ^ currentpoint_inside_clipedge

            if prevpoint_inside_clipedge:
                outputList.append( intersect(prev_point, current_point, polyB[j], polyB[k]) )
            if currentpoint_inside_clipedge:
                outputList.append( current_point )
                
            prev_point = current_point
                
        inputList, outputList = outputList, []        
        inputListSize = len(inputList)        
        j = k

    return inputList

def polygon_area(poly):   
    w = 0    
    i = len(poly) - 2
    j = len(poly) - 1
    for k in range(len(poly)):        
        w += poly[j][0] * (poly[k][1] - poly[i][1])        
        i = j
        j = k        
    area = w * 0.5
    return area

def rasterize_polygon(poly, cube_width, cube_depth, out=None):    
    if out is None:
         out = torch.zeros( (cube_width * cube_depth) )
    
    for y in range(cube_depth):
        for x in range(cube_width):
            pixel = [
                    (x + 0, y + 0),
                    (x + 1, y + 0),
                    (x + 1, y + 1),
                    (x + 0, y + 1)
                ]            
            intersect = polygon_intersect_convexpolygon(poly, pixel)
            out[y * cube_width + x] = polygon_area(intersect)    
            
    return out


def make_slice_matrix(xray_width, xray_count, xray_width_scale, xray_radians_step, cube_width, cube_depth):
    
    xray_slice_matrix = torch.zeros(  ( xray_count * xray_width, cube_width * cube_depth) )

    for ixray in range(xray_count):
        
        #angle of this xray 
        screen_surface_radians = ixray * xray_radians_step
        
        #vector describing the screen position
        screen_surface_x1 = math.cos(screen_surface_radians) * 0.5 * xray_width_scale * xray_width
        screen_surface_y1 = math.sin(screen_surface_radians) * 0.5 * xray_width_scale * xray_width
        screen_surface_x2 = -screen_surface_x1
        screen_surface_y2 = -screen_surface_y1

        screen_surface_x1 += xray_width * 0.5
        screen_surface_y1 += xray_width * 0.5
        screen_surface_x2 += xray_width * 0.5
        screen_surface_y2 += xray_width * 0.5
        
        #vector perpendicular to screen
        perp_vector_x = screen_surface_y1 - screen_surface_y2
        perp_vector_y = screen_surface_x2 - screen_surface_x1
        
        for ix in range(xray_width):
            pcnt = 100 * (ixray * xray_width + ix) / (xray_count * xray_width)
            print('constructing kernel matrix %.2f%%' % (pcnt,), end='\r')
            
            #vector describing the pixel position
            pixel_x1 = (screen_surface_x2 - screen_surface_x1) * (ix / xray_width) + screen_surface_x1
            pixel_x2 = (screen_surface_x2 - screen_surface_x1) * ((ix + 1) / xray_width) + screen_surface_x1
            pixel_y1 = (screen_surface_y2 - screen_surface_y1) * (ix / xray_width) + screen_surface_y1
            pixel_y2 = (screen_surface_y2 - screen_surface_y1) * ((ix + 1) / xray_width) + screen_surface_y1
            
            #polygon describing the ray cast from this pixel
            poly = [
                    (pixel_x1 - perp_vector_x, pixel_y1 - perp_vector_y),
                    (pixel_x2 - perp_vector_x, pixel_y2 - perp_vector_y),
                    (pixel_x2 + perp_vector_x, pixel_y2 + perp_vector_y),
                    (pixel_x1 + perp_vector_x, pixel_y1 + perp_vector_y)
                ]
            
            #draw the ray into xray_slice_matrix
            #describing how this pixel interacts with a horizontal slice of the cube
            rasterize_polygon(poly, cube_width, cube_depth, xray_slice_matrix[ixray * xray_width + ix])
    print('')
    return xray_slice_matrix

def make_cylinder_mask(cube_depth, cube_width):
    mask = torch.zeros((cube_depth, cube_width))
    
    hd = cube_depth // 2
    hw = cube_width // 2
    r2 = hw * hw
    for y in range(cube_depth):
        for x in range(cube_width):
            cx = (x - hw) + 0.5
            cy = (y - hd) + 0.5
            if cx * cx + cy * cy <= r2:
                mask[y,x] = 1.0  
    return mask


xray_width = 16
xray_height = 16    # must equal cube_height
xray_count = 24     # these numbers usually >= the cube numbers, for more accuracy

cube_width = 16
cube_height = 16    # must equal xray_height
cube_depth = 16

# scale * sqrt(2) would scale to cover the entire cube 
# use a cylinder mask instead to solve for the covered cylinder
# also correct for different pixel widths
xray_width_scale = 1.0 * (cube_width / xray_width)
use_cylinder_mask = True

# each xray is taken at a different angle on the cylinder
xray_radians_step = (torch.pi * 1) / xray_count


xray_slice_matrix = make_slice_matrix(xray_width, xray_count, xray_width_scale, xray_radians_step, cube_width, cube_depth)

if use_cylinder_mask:
    cylinder_mask = make_cylinder_mask(cube_depth, cube_width)    
    xray_slice_matrix *= cylinder_mask.reshape(1,-1)

print('inverting kernel matrix')
xray_slice_matrix_pinv = torch.linalg.pinv(xray_slice_matrix)


cube_truth = torch.rand( (cube_height, cube_depth, cube_width) )

if use_cylinder_mask:  
    cube_truth *= cylinder_mask

# simulate xrays by applying xray_slice_matrix to cube_truth
xrays = xray_slice_matrix @ cube_truth.reshape(cube_height, -1, 1)
xrays = xrays.reshape(xray_height, xray_count, xray_width)

# use 2d xrays to predict contents of 3d cube
cube_pred = xray_slice_matrix_pinv @ xrays.reshape(xray_height, -1, 1)
cube_pred = cube_pred.reshape(cube_height, cube_depth, cube_width)

print('predict cube from xrays (CT scan, xray computed tomography)')
print( 'cube_pred max error', torch.amax(torch.abs(cube_pred - cube_truth)) )
print( 'cube_pred mean error', torch.mean(torch.abs(cube_pred - cube_truth)) )
print( 'cube_pred min error', torch.amin(torch.abs(cube_pred - cube_truth)) )


# there is no dc_ray with a cancellation function
# this is just a demonstration that the math works in reverse

# each dc_ray value * its kernel delivers energy
# sum of slices = energy delivered
# energy = xray_slice_matrix.T @ dc_rays[0].reshape(-1)

energy_truth = torch.rand( (cube_height, cube_depth, cube_width) )
if use_cylinder_mask:  
    energy_truth *= cylinder_mask

# predict dc_rays required to deliver energy_truth
dc_rays = xray_slice_matrix_pinv.T @ energy_truth.reshape(cube_height, -1, 1)
dc_rays = dc_rays.reshape(xray_height, xray_count, xray_width)

# simulate delivering dc_rays
energy_pred = xray_slice_matrix.T @ dc_rays.reshape(xray_height, -1, 1)
energy_pred = energy_pred.reshape(cube_height, cube_depth, cube_width)

print('predict dc_rays to minimize(energy_pred - energy_truth)')
print( 'energy_pred max error', torch.amax(torch.abs(energy_pred - energy_truth)) )
print( 'energy_pred mean error', torch.mean(torch.abs(energy_pred - energy_truth)) )
print( 'energy_pred min error', torch.amin(torch.abs(energy_pred - energy_truth)) )




# for proper handling of time
# something similar to fft_reverse_tomography.py could work
# xray_slice_matrix becomes the falloff matrix

