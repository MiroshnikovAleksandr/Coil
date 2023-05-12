# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 12:28:30 2021

@author: Pavel

Functions for magnetic field calculation
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import ellipk,ellipkm1, ellipe

# Circle equasion
def dist(x1, y1, x2, y2):
    """
    Return the distance between two points A(x1,y1) and B(x2,y2)

    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Cirqular mask for COV calculation
def mask_circle(tiles, cx, cy, r):
    """
    Creates circular binary mask in array "tiles" with center in point C(cx,cy)
    and radius "r"
      
    """
    for x in range(cx - r, cx + r):
        for y in range(cy - r, cy + r):
            if dist(cx, cy, x, y) <= r:
                tiles[x][y] = 1

# COV for cirqular coil
def COV_circ(Bz,max_coil_r,height,spacing):
    """
    Coefficient of variation calculation for circular coil
    
    Parameters
    ----------
    Bz : 
        Field for CoV calculation
    max_coil_r : 
        Maximum coil radius [m]
    height : 
        Height above the coil
    spacing : 
        Spacing between coil and the calculation domain boundary
      
    """
    calc_radius = max_coil_r*spacing # Calculation domain length
    
    view_line = height/(2*calc_radius/len(Bz)) + len(Bz)/2 
    view_line = int(view_line)
    
    cp = len(Bz) # Calculation domain
    cx = cp // 2 # center of calc area
    cy = cp // 2
    cell_size = calc_radius/cp
    tiles = np.zeros([cp,cp])
    r_cov_m = max_coil_r*0.9 # Uniform area
    r_cov = r_cov_m/cell_size/2 # Uniform area in cells
    mask_circle(tiles, cx, cy, round(r_cov)) 
    Bz_masked = np.multiply(Bz[:,:,view_line],tiles)
    B_mean = np.sum(Bz_masked)/np.sum(tiles)
    B_std = np.sqrt(np.sum((Bz_masked-np.multiply(B_mean,tiles))**2)/np.sum(tiles))
    COV = B_std/B_mean
    return COV


def Bz(a_max, a_min, n, I, spacing, cp, r_i):
    """
    Calculate Bz field of circular multiturn coil
    
    Parameters
    ----------
    a_max : 
        Maximum coil radius [m]
    a_min : 
        Minimum coil radius [m]
    n : integer
        Number of coil turns 
    I : 
        Current [A]
    spacing : 
        Spacing between coil and the calculation domain boundary
    cp : 
        Calculation domain points
    r_i : array
        Array of turns radius
          
    """ 
    mu0 = np.pi*4e-7 # Vacuum permeability
    calc_radius = a_max*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv,yv,zv = np.meshgrid(x,x,x) # Creating meshgrid
    
    Bz_sum = np.zeros((cp,cp,cp)) # Preallocation
    
    ro      = np.sqrt(xv**2+yv**2)
    r       = np.sqrt(xv**2+yv**2+zv**2)
    C       = mu0*I/np.pi
    
    for i, radius in enumerate(r_i):
        alpha   = np.sqrt(radius**2+r**2-2*radius*ro)
        beta    = np.sqrt(radius**2+r**2+2*radius*ro)
        k       = np.sqrt(1-alpha**2/beta**2)
    
        Bz_i = C/(2*alpha**2*beta)*((radius**2-r**2)*ellipe(k**2)+alpha**2*ellipk(k**2))
        Bz_sum = Bz_sum + Bz_i
    
    return Bz_sum


def Bx(a_max,a_min,n,I,spacing,cp,x_i):
    """
    Calculate Bx field of circular multiturn coil
    
    Parameters
    ----------
    a_max : 
        Maximum coil radius [m]
    a_min : 
        Minimum coil radius [m]
    n : integer
        Number of coil turns 
    I : 
        Current [A]
    spacing : 
        Spacing between coil and the calculation domain boundary
    cp : 
        Calculation domain points
    r_i : array
        Array of turns radius
      
    """
    mu0 = np.pi*4e-7
    calc_radius = a_max*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv,yv,zv = np.meshgrid(x,x,x) # Creating meshgrid
    
    Bx_sum = np.zeros((cp,cp,cp)) # Preallocation
    
    ro      = np.sqrt(xv**2+yv**2)
    r       = np.sqrt(xv**2+yv**2+zv**2)
    C       = mu0*I/np.pi
    
    for i,radius in enumerate(x_i):
        alpha   = np.sqrt(radius**2+r**2-2*radius*ro)
        beta    = np.sqrt(radius**2+r**2+2*radius*ro)
        k       = np.sqrt(1-alpha**2/beta**2)
    
        Bx_i = C*xv*zv/(2*alpha**2*beta*ro**2)*((radius**2+r**2)*ellipe(k**2)-alpha**2*ellipk(k**2))
        Bx_sum = Bx_sum + Bx_i
            
    return Bx_sum

def By(a_max,a_min,n,I,spacing,cp,x_i):
    """
    Calculate By field of circular multiturn coil
    
    Parameters
    ----------
    a_max : 
        Maximum coil radius [m]
    a_min : 
        Minimum coil radius [m]
    n : integer
        Number of coil turns 
    I : 
        Current [A]
    spacing : 
        Spacing between coil and the calculation domain boundary
    cp : 
        Calculation domain points
    r_i : array
        Array of turns radius
      
    """
    mu0 = np.pi*4e-7
    calc_radius = a_max*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv,yv,zv = np.meshgrid(x,x,x) # Creating meshgrid
    
    By_sum = np.zeros((cp,cp,cp)) # Preallocation
    
    ro      = np.sqrt(xv**2+yv**2)
    r       = np.sqrt(xv**2+yv**2+zv**2)
    C       = mu0*I/np.pi
    
    for i,radius in enumerate(x_i):
        alpha   = np.sqrt(radius**2+r**2-2*radius*ro)
        beta    = np.sqrt(radius**2+r**2+2*radius*ro)
        k       = np.sqrt(1-alpha**2/beta**2)
           
        By_i = C*yv*zv/(2*alpha**2*beta*ro**2)*((radius**2+r**2)*ellipe(k**2)-alpha**2*ellipk(k**2))
        By_sum = By_sum + By_i
    
    return By_sum

def B_vector(a_max,a_min,n,I,spacing,cp,x_i):
    """
    Calculate vector B field of circular multiturn coil
    
    Parameters
    ----------
    a_max : 
        Maximum coil radius [m]
    a_min : 
        Minimum coil radius [m]
    n : integer
        Number of coil turns 
    I : 
        Current [A]
    spacing : 
        Spacing between coil and the calculation domain boundary
    cp : 
        Calculation domain points
    r_i : array
        Array of turns radius
      
    """
    mu0 = np.pi*4e-7
    calc_radius = a_max*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv,yv,zv = np.meshgrid(x,x,x) # Creating meshgrid
    
    Bz_sum = np.zeros((cp,cp,cp)) # Preallocation
    Bx_sum = np.zeros((cp,cp,cp)) 
    By_sum = np.zeros((cp,cp,cp)) 
    
    ro      = np.sqrt(xv**2+yv**2)
    r       = np.sqrt(xv**2+yv**2+zv**2)
    C       = mu0*I/np.pi
    
    for i,radius in enumerate(x_i):
        alpha   = np.sqrt(radius**2+r**2-2*radius*ro)
        beta    = np.sqrt(radius**2+r**2+2*radius*ro)
        k       = np.sqrt(1-alpha**2/beta**2)
    
        Bx_i = C*xv*zv/(2*alpha**2*beta*ro**2)*((radius**2+r**2)*ellipe(k**2)-alpha**2*ellipk(k**2))
        Bx_sum = Bx_sum + Bx_i
        
        By_i = C*yv*zv/(2*alpha**2*beta*ro**2)*((radius**2+r**2)*ellipe(k**2)-alpha**2*ellipk(k**2))
        By_sum = By_sum + By_i
        
        Bz_i = C/(2*alpha**2*beta)*((radius**2-r**2)*ellipe(k**2)+alpha**2*ellipk(k**2))
        Bz_sum = Bz_sum + Bz_i
    
    B_vector = np.array([Bx_sum,By_sum,Bz_sum])
    return B_vector

def Bz_single(a,I,spacing,cp):
    """
    Calculate Bz field of sincle-turn circular turn
    
    Parameters
    ----------
    a : 
        Turn radius [m]
    I : 
        Current [A]
    spacing : 
        Spacing between coil and the calculation domain boundary
    cp : 
        Calculation domain points
      
    """
    mu0 = np.pi*4e-7
    calc_radius = a*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv,yv,zv = np.meshgrid(x,x,x) # Creating meshgrid
    
    Bz = np.zeros((cp,cp,cp)) # Preallocation
    
    ro      = np.sqrt(xv**2+yv**2)
    r       = np.sqrt(xv**2+yv**2+zv**2)
    C       = mu0*I/np.pi
    alpha   = np.sqrt(a**2+r**2-2*a*ro)
    beta    = np.sqrt(a**2+r**2+2*a*ro)
    k       = np.sqrt(1-alpha**2/beta**2)
        
    Bz = C/(2*alpha**2*beta)*((a**2-r**2)*ellipe(k**2)+alpha**2*ellipk(k**2))
            
    return Bz


def Bz_square_single(m,n,I,spacing,cp):
    """
    Calculate Bz field of sincle-turn circular turn
    
    Parameters
    ----------
    m : 
        Side length x [m]
    n : 
        Side length y [m]
    I : 
        Current [A]
    spacing : 
        Spacing between coil and the calculation domain boundary
    cp : 
        Calculation domain points
      
    """
    mu0 = np.pi*4e-7
    calc_radius = np.amax([m,n])*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv,yv,zv = np.meshgrid(x,x,x) # Creating meshgrid
    
    C = mu0*I/(4*np.pi)

    c1 = xv + m/2
    c2 = xv - m/2
    c3 = xv - m/2
    c4 = xv + m/2
    
    d1 = yv + n/2
    d2 = yv + n/2
    d3 = yv - n/2
    d4 = yv - n/2
    
    r1 = np.sqrt(c1**2+d1**2+zv**2)
    r2 = np.sqrt(c2**2+d2**2+zv**2)
    r3 = np.sqrt(c3**2+d3**2+zv**2)
    r4 = np.sqrt(c4**2+d4**2+zv**2)
  
    Bz_square = C*((c2/r2/(r2+d2))+(d2/r2/(r2+c2))-(c1/r1/(r1+d1)+d1/r1/(r1+c1))+(c4/r4/(r4+d4)+d4/r4/(r4+c4))-(c3/r3/(r3+d3)+d3/r3/(r3+c3)))
            
    return Bz_square


def Bz_square(m_max,n_max,n,I,spacing,cp,m_i,n_i):
    """
    Calculate Bz field of rectangular multiturn coil
    
    Parameters
    ----------
    m_max : 
        Maximum x-side length [m]
    n_max : 
        Maximum y-side length [m]
    n : integer
        Number of coil turns 
    I : 
        Current [A]
    spacing : 
        Spacing between coil and the calculation domain boundary
    cp : 
        Calculation domain points
    r_i : array
        Array of turns radius
          
    """ 
    

    calc_radius = np.amax([m_max,n_max])*spacing # Calculation domain length
 
    
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv,yv,zv = np.meshgrid(x,x,x) # Creating meshgrid
    
    Bz_square_sum = np.zeros((cp,cp,cp)) # Preallocation
    mu0 = np.pi*4e-7 # Vacuum permeability

    C = mu0*I/(4*np.pi)

    for i, size_m in enumerate(m_i):
        for j, size_n in enumerate(n_i):
            if i == j:
                c1 = xv + size_m/2
                c2 = xv - size_m/2
                c3 = xv - size_m/2
                c4 = xv + size_m/2
                
                d1 = yv + size_n/2
                d2 = yv + size_n/2
                d3 = yv - size_n/2
                d4 = yv - size_n/2
                
                r1 = np.sqrt(c1**2+d1**2+zv**2)
                r2 = np.sqrt(c2**2+d2**2+zv**2)
                r3 = np.sqrt(c3**2+d3**2+zv**2)
                r4 = np.sqrt(c4**2+d4**2+zv**2)
                
                Bz_square_i = C*((c2/r2/(r2+d2))+(d2/r2/(r2+c2))-(c1/r1/(r1+d1)+d1/r1/(r1+c1))+(c4/r4/(r4+d4)+d4/r4/(r4+c4))-(c3/r3/(r3+d3)+d3/r3/(r3+c3)))
    
                Bz_square_sum = Bz_square_sum + Bz_square_i
              
    return Bz_square_sum


def plot_2d(Bz,height,a_max,spacing,cp):
    """
    
      
    """
    calc_radius = a_max*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    view_line = height/(2*calc_radius/np.size(x)) + np.size(x)/2 
    view_line = int(view_line)
    fig = plt.figure()
    plt.plot(x*1e2,Bz[:,view_line,view_line]*1e6)
    plt.xlabel('x [cm]')
    plt.ylabel('Bz [uT]')
    plt.title('Bz Field at {} mm height'.format(height*1e3))
  
def plot_3d(Bz,height,a_max,spacing,cp):
    """
    
      
    """
    calc_radius = a_max*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv,yv,zv = np.meshgrid(x,x,x) # Creating meshgrid
    view_line = height/(2*calc_radius/np.size(x)) + np.size(x)/2 
    view_line = int(view_line)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xv[:,:,1]*1e2,yv[:,:,1]*1e2,Bz[:,:,view_line]*1e6, cmap='inferno')
    ax.set_title('Bz Field at {} mm height'.format(height*1e3))
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')    


def plot_vector(Bz,height,a_max,spacing,cp):
    """
    
      
    """
    calc_radius = a_max*spacing # Calculation domain length
    x = np.linspace(-calc_radius,calc_radius,cp) 
    xv, yv, zv = np.meshgrid(x,x,x) # Creating meshgrid
    view_line = height/(2*calc_radius/np.size(x)) + np.size(x)/2 
    view_line = int(view_line)
    fig = plt.figure()
    
    ax = fig.add_subplot(projection='3d')
    ax.quiver(xv[::30,::30,::30],yv[::30,::30,::30],zv[::30,::30,::30],Bx_sum[::30,::30,::30],By_sum[::30,::30,::30],Bz_sum[::30,::30,::30], length=0.03, normalize=True)

    ax.set_title('Vector field'.format(height*1e3))    
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')   
    plt.show()

def plot_coil(a_max,spacing,r_i):
    """
    
      
    """
    fig = plt.figure(figsize=(3, 3), dpi=300)
    
    
    ax = fig.subplots()
    # ax.set_xlim((0, 2*a_max*spacing))
    # ax.set_ylim((0, 2*a_max*spacing))
    ax.set_xlim((-a_max*spacing, a_max*spacing))
    ax.set_ylim((-a_max*spacing, a_max*spacing))
    for i,radius in enumerate(r_i):
        # circle = plt.Circle((a_max*spacing, a_max*spacing), radius, fill=False)
        circle = plt.Circle((0, 0), radius, fill=False)

        ax.add_patch(circle)
        
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        
    plt.show()

def plot_square_coil(m_max,n_max,spacing,m_i,n_i):
    """
    
      
    """
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.subplots()
    max_size = np.max([m_max,n_max])*spacing

    ax.set_xlim((-max_size, max_size))
    ax.set_ylim((-max_size, max_size))
    
    for i,size_m in enumerate(m_i):
        for j,size_n in enumerate(n_i):
            if i==j:
                rec = plt.Rectangle((-size_m/2,-size_n/2),size_m,size_n,fill=False)
                ax.add_patch(rec)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        
    plt.show()