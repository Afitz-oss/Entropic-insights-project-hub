import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import random
import time

# Global variables
grid_resolution = 35
current_mode = 'realistic'
masses = [8, 2, 0.8]  # Sun, Earth, Moon
positions = [(-1, -1), (4, 2), (5.5, 3)]
animation_active = False

def spacetime_curvature(x, y, masses, positions):
    """Enhanced spacetime curvature calculation with gentler curves"""
    z = np.zeros_like(x, dtype=float)
    
    for mass, pos in zip(masses, positions):
        dx = x - pos[0]
        dy = y - pos[1]
        r = np.sqrt(dx**2 + dy**2 + 0.1)
        
        # Gentler curvature - reduced the depth significantly
        if mass == masses[0]:  # Sun - make it less deep
            curvature_strength = 0.15  # Much gentler for sun
        else:  # Earth and Moon
            curvature_strength = 0.25
            
        z = z - (mass * curvature_strength) / (r**1.1 + 1.5)
    
    return z

def tidal_forces(x, y, masses, positions):
    """Calculate tidal force effects for enhanced realism"""
    tidal = np.zeros_like(x)
    for mass, pos in zip(masses, positions):
        dx = x - pos[0]
        dy = y - pos[1]
        r = np.sqrt(dx**2 + dy**2 + 0.1)
        tidal += mass * 0.05 * (dx**2 + dy**2) / (r**4 + 1.0)
    return tidal

def create_enhanced_stars(n_stars=200, mode='realistic'):
    """Create enhanced star field with different modes"""
    modes = {
        'realistic': {'colors': ['white', 'lightblue', 'yellow'], 'sizes': (5, 30)},
        'artistic': {'colors': ['magenta', 'cyan', 'yellow', 'white'], 'sizes': (3, 40)},
        'scientific': {'colors': ['white', 'lightgray'], 'sizes': (2, 15)},
        'nebula': {'colors': ['purple', 'pink', 'blue', 'white'], 'sizes': (10, 50)}
    }
    
    settings = modes.get(mode, modes['realistic'])
    stars_data = []
    
    for _ in range(n_stars):
        x = random.uniform(-15, 15)
        y = random.uniform(-15, 15)
        z = random.uniform(2, 8)
        size = random.uniform(*settings['sizes'])
        color = random.choice(settings['colors'])
        alpha = random.uniform(0.4, 1.0)
        
        stars_data.append({'pos': [x, y, z], 'size': size, 'color': color, 'alpha': alpha})
    
    return stars_data

def create_textured_sphere(ax, radius, position, color_scheme, surface_z, resolution=50):
    """Create enhanced spheres with texture and better lighting"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v)) + position[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + position[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + surface_z + radius
    
    # Create texture pattern
    texture = np.sin(6*u)[:, np.newaxis] * np.sin(4*v)[np.newaxis, :] * 0.1
    
    if color_scheme == 'sun':
        colors = plt.cm.YlOrRd(0.7 + texture)
    elif color_scheme == 'earth':
        # Create beautiful blue Earth with cloud-like variations
        blue_base = np.array([0.1, 0.5, 0.9, 1.0])  # Ocean blue
        green_land = np.array([0.2, 0.6, 0.2, 1.0])  # Land green
        # Mix blue oceans with green land based on texture
        land_mask = texture > 0.05
        colors = np.where(land_mask[..., np.newaxis], green_land, blue_base)
    elif color_scheme == 'moon':
        colors = plt.cm.gray(0.5 + texture)
    else:
        colors = color_scheme
    
    return ax.plot_surface(x, y, z, facecolors=colors, alpha=0.95, shade=True,
                          lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=60))

def create_enhanced_spacetime_visualization():
    """Enhanced spacetime visualization with better celestial body positioning"""
    global fig, ax, stars_data
    
    # Create figure with better styling
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    
    # Higher resolution grid
    x_range = np.linspace(-12, 12, grid_resolution)
    y_range = np.linspace(-12, 12, grid_resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate enhanced spacetime curvature
    Z = spacetime_curvature(X, Y, masses, positions)
    
    # Add tidal effects for more realism (but gentler)
    tidal = tidal_forces(X, Y, masses, positions)
    Z += tidal * 0.3  # Reduce tidal effect impact
    
    # Enhanced wireframe with gradient colors and varying opacity
    colors = plt.cm.plasma(np.linspace(0, 1, len(x_range)))
    
    for i in range(len(x_range)):
        # Distance-based alpha for depth effect
        alpha = 0.2 + 0.6 * (1 - abs(i - len(x_range)//2) / (len(x_range)//2))
        ax.plot(X[i, :], Y[i, :], Z[i, :], color=colors[i], alpha=alpha, linewidth=1.0)
    
    for j in range(len(y_range)):
        alpha = 0.2 + 0.6 * (1 - abs(j - len(y_range)//2) / (len(y_range)//2))
        ax.plot(X[:, j], Y[:, j], Z[:, j], color=colors[j], alpha=alpha, linewidth=1.0)
    
    # Enhanced celestial bodies with better positioning
    sun_surface_z = spacetime_curvature(np.array([[positions[0][0]]]), np.array([[positions[0][1]]]), masses, positions)[0,0]
    earth_surface_z = spacetime_curvature(np.array([[positions[1][0]]]), np.array([[positions[1][1]]]), masses, positions)[0,0]
    moon_surface_z = spacetime_curvature(np.array([[positions[2][0]]]), np.array([[positions[2][1]]]), masses, positions)[0,0]
    
    # Add elevation offsets to keep bodies visible above the grid
    sun_elevation = max(0.5, -sun_surface_z * 0.3)  # Keep sun well above grid
    earth_elevation = max(0.2, -earth_surface_z * 0.5)
    moon_elevation = max(0.1, -moon_surface_z * 0.7)
    
    create_textured_sphere(ax, 1.2, positions[0], 'sun', sun_surface_z + sun_elevation, 60)
    create_textured_sphere(ax, 0.3, positions[1], 'earth', earth_surface_z + earth_elevation, 40)
    create_textured_sphere(ax, 0.08, positions[2], 'moon', moon_surface_z + moon_elevation, 30)
    
    # Enhanced background with mode-based stars
    stars_data = create_enhanced_stars(250, current_mode)
    
    for star in stars_data:
        ax.scatter(*star['pos'], c=star['color'], s=star['size'], 
                  alpha=star['alpha'], marker='*', edgecolors='white', linewidths=0.1)
    
    # Add educational labels and information
    add_educational_features()
    
    # Enhanced styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.grid(False)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Set limits and view
    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])
    ax.set_zlim([-3, 4])  # Adjusted z-limits for better visibility
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    ax.view_init(elev=25, azim=45)
    
    # Enhanced title with physics equation
    plt.suptitle('Spacetime Curvature: Einstein\'s General Relativity\n' + 
                'Gμν = 8πTμν  •  Sun, Earth, and Moon System', 
                color='white', fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    return fig, ax

def add_educational_features():
    """Add educational labels with corrected positioning"""
    
    # Calculate surface positions for labels with elevation adjustments
    sun_surface_z = spacetime_curvature(np.array([[positions[0][0]]]), np.array([[positions[0][1]]]), masses, positions)[0,0]
    earth_surface_z = spacetime_curvature(np.array([[positions[1][0]]]), np.array([[positions[1][1]]]), masses, positions)[0,0]
    moon_surface_z = spacetime_curvature(np.array([[positions[2][0]]]), np.array([[positions[2][1]]]), masses, positions)[0,0]
    
    # Apply same elevation adjustments as spheres
    sun_elevation = max(0.5, -sun_surface_z * 0.3)
    earth_elevation = max(0.2, -earth_surface_z * 0.5)
    moon_elevation = max(0.1, -moon_surface_z * 0.7)
    
    # Enhanced labels with corrected positions
    ax.text(positions[0][0], positions[0][1], sun_surface_z + sun_elevation + 2.5, 
            'SUN ☉\nMass: 1.989×10³⁰ kg\nRadius: 696,340 km', 
            color='gold', fontsize=11, ha='center', va='bottom', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.text(positions[1][0], positions[1][1], earth_surface_z + earth_elevation + 0.8, 
            'EARTH 🌍\nMass: 5.972×10²⁴ kg\nRadius: 6,371 km\n(Enhanced for visibility)', 
            color='lightblue', fontsize=10, ha='center', va='bottom', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.text(positions[2][0], positions[2][1], moon_surface_z + moon_elevation + 0.4, 
            'MOON 🌙\nMass: 7.342×10²² kg\nRadius: 1,737 km\n(Enhanced for visibility)', 
            color='silver', fontsize=9, ha='center', va='bottom', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Curvature measurement indicators with adjusted positioning
    elevations = [sun_elevation, earth_elevation, moon_elevation]
    for i, (mass, pos, elevation) in enumerate(zip(masses, positions, elevations)):
        surface_z = spacetime_curvature(np.array([[pos[0]]]), np.array([[pos[1]]]), masses, positions)[0,0]
        actual_surface = surface_z + elevation
        
        # Curvature depth line from grid to actual surface
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [surface_z, actual_surface], 
               color='red', linewidth=2, alpha=0.8, linestyle='--')
        
        # Depth measurement
        ax.text(pos[0] + 0.8, pos[1], (surface_z + actual_surface)/2, 
               f'Curvature: {abs(surface_z):.2f}', 
               color='red', fontsize=9, weight='bold')
    
    # Physics information panel with updated curvature values
    curvatures = [
        abs(spacetime_curvature(np.array([[pos[0]]]), np.array([[pos[1]]]), masses, positions)[0,0])
        for pos in positions
    ]
    
    info_text = f"""🌌 SPACETIME METRICS
    
📐 Curvature Depths:
• Sun: {curvatures[0]:.3f} (gentler)
• Earth: {curvatures[1]:.3f}  
• Moon: {curvatures[2]:.3f}

🪐 Size Visualization:
• Sun: 696,340 km (reference size)
• Earth: 12,742 km (enhanced for visibility)
• Moon: 3,474 km (enhanced for visibility)
• Maintains relative proportions!

🧮 Einstein Field Equation:
Gμν = 8πTμν

💭 "Spacetime tells matter how to move;
   matter tells spacetime how to curve"
   - John Wheeler

⚡ Current Mode: {current_mode.upper()}
✨ NEW: Realistic planetary scales & Blue Earth!
"""
    
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
             color='white', fontsize=10, va='top', ha='left', weight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='navy', alpha=0.8))

def create_orbital_animation():
    """Create animated orbital motion"""
    global animation_active, anim
    
    if animation_active:
        return
    
    print("🎬 Starting orbital animation...")
    animation_active = True
    
    # Store original positions
    original_positions = positions.copy()
    
    def animate_orbits(frame):
        global positions
        
        t = frame * 0.05  # Animation speed
        
        # Earth orbit around Sun (elliptical)
        earth_orbit_a = 5.5  # Semi-major axis
        earth_orbit_b = 4.8  # Semi-minor axis
        earth_angle = t * 0.3
        
        new_earth_pos = [
            original_positions[0][0] + earth_orbit_a * np.cos(earth_angle),
            original_positions[0][1] + earth_orbit_b * np.sin(earth_angle)
        ]
        
        # Moon orbit around Earth
        moon_orbit_radius = 1.8
        moon_angle = t * 1.5  # Moon orbits faster
        
        new_moon_pos = [
            new_earth_pos[0] + moon_orbit_radius * np.cos(moon_angle),
            new_earth_pos[1] + moon_orbit_radius * np.sin(moon_angle)
        ]
        
        # Update global positions
        positions[1] = new_earth_pos
        positions[2] = new_moon_pos
        
        # Clear and redraw
        ax.clear()
        create_enhanced_spacetime_visualization()
        
        # Add orbital trails
        earth_trail_x = [original_positions[0][0] + earth_orbit_a * np.cos(t * 0.3 - i * 0.1) for i in range(20)]
        earth_trail_y = [original_positions[0][1] + earth_orbit_b * np.sin(t * 0.3 - i * 0.1) for i in range(20)]
        earth_trail_z = [spacetime_curvature(np.array([[x]]), np.array([[y]]), masses, positions)[0,0] + 0.5 
                        for x, y in zip(earth_trail_x, earth_trail_y)]
        
        ax.plot(earth_trail_x, earth_trail_y, earth_trail_z, 'b-', alpha=0.5, linewidth=2)
        
        return ax.collections + ax.lines
    
    anim = FuncAnimation(fig, animate_orbits, frames=300, interval=100, blit=False, repeat=True)
    return anim

def add_advanced_controls():
    """Enhanced interactive controls with real-time parameter adjustment"""
    
    def on_key_advanced(event):
        global masses, grid_resolution, current_mode, animation_active
        
        if event.key == 'left':
            ax.azim -= 5
        elif event.key == 'right':
            ax.azim += 5
        elif event.key == 'up':
            ax.elev += 5
        elif event.key == 'down':
            ax.elev -= 5
        elif event.key == 'r':
            ax.view_init(elev=25, azim=45)
        elif event.key == '1':
            masses[0] += 1
            print(f"☉ Sun mass increased to {masses[0]}")
            update_visualization()
        elif event.key == '2':
            masses[0] = max(1, masses[0] - 1)
            print(f"☉ Sun mass decreased to {masses[0]}")
            update_visualization()
        elif event.key == '3':
            masses[1] += 0.5
            print(f"🌍 Earth mass increased to {masses[1]}")
            update_visualization()
        elif event.key == '4':
            masses[1] = max(0.1, masses[1] - 0.5)
            print(f"🌍 Earth mass decreased to {masses[1]}")
            update_visualization()
        elif event.key == 'g':
            toggle_grid_resolution()
        elif event.key == 'm':
            cycle_visual_mode()
        elif event.key == 'a':
            toggle_animation()
        elif event.key == 's':
            save_screenshot()
        elif event.key == 'e':
            export_data()
        elif event.key == 'v':
            export_animation()
        elif event.key == 'h':
            show_help()
        elif event.key == 'escape':
            plt.close('all')
        
        plt.draw()
    
    def update_visualization():
        """Redraw with new parameters"""
        ax.clear()
        create_enhanced_spacetime_visualization()
    
    def toggle_grid_resolution():
        """Switch between high and low resolution"""
        global grid_resolution
        grid_resolution = 25 if grid_resolution == 35 else 35
        print(f"📊 Grid resolution: {grid_resolution}x{grid_resolution}")
        update_visualization()
    
    def cycle_visual_mode():
        """Cycle through visual modes"""
        global current_mode
        modes = ['realistic', 'artistic', 'scientific', 'nebula']
        current_idx = modes.index(current_mode)
        current_mode = modes[(current_idx + 1) % len(modes)]
        print(f"🎨 Visual mode: {current_mode.upper()}")
        update_visualization()
    
    def toggle_animation():
        """Start/stop orbital animation"""
        global animation_active
        if not animation_active:
            create_orbital_animation()
            print("▶️ Animation started")
        else:
            animation_active = False
            if 'anim' in globals():
                anim.event_source.stop()
            print("⏸️ Animation stopped")
    
    def save_screenshot():
        """Save high-quality screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"spacetime_curvature_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black', 
                   edgecolor='none', pad_inches=0.1)
        print(f"📸 Screenshot saved: {filename}")
    
    def export_data():
        """Export spacetime data as CSV"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"spacetime_data_{timestamp}.csv"
        
        x_range = np.linspace(-12, 12, 50)
        y_range = np.linspace(-12, 12, 50)
        X, Y = np.meshgrid(x_range, y_range)
        Z = spacetime_curvature(X, Y, masses, positions)
        
        data = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        header = f"X,Y,Curvature\n# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n# Masses: {masses}\n# Positions: {positions}"
        
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
        print(f"💾 Data exported: {filename}")
    
    def export_animation():
        """Export animation as GIF"""
        print("🎬 Creating animation export... This may take a few minutes.")
        try:
            anim = create_orbital_animation()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"spacetime_animation_{timestamp}.gif"
            
            # Use pillow writer for GIF
            writer = plt.matplotlib.animation.PillowWriter(fps=10)
            anim.save(filename, writer=writer)
            print(f"🎥 Animation exported: {filename}")
        except Exception as e:
            print(f"❌ Animation export failed: {e}")
    
    def show_help():
        """Display comprehensive help"""
        help_text = """
🌌 ENHANCED SPACETIME CURVATURE VISUALIZER 🌌

🎮 NAVIGATION CONTROLS:
  Mouse:      Rotate and zoom view
  ← → ↑ ↓:   Fine rotation control  
  R:          Reset to original view
  ESC:        Exit program

🔧 PHYSICS CONTROLS:
  1 / 2:      Increase/decrease Sun mass
  3 / 4:      Increase/decrease Earth mass
  
🎨 VISUAL CONTROLS:
  G:          Toggle grid resolution (25x25 ↔ 35x35)
  M:          Cycle visual modes (realistic→artistic→scientific→nebula)
  
🎬 ANIMATION CONTROLS:
  A:          Start/stop orbital animation
  
💾 EXPORT CONTROLS:
  S:          Save high-quality screenshot (PNG)
  E:          Export spacetime data (CSV)
  V:          Export orbital animation (GIF)
  
❓ HELP:
  H:          Show this help menu

🌟 FEATURES:
  • Real-time mass adjustment affects curvature
  • Multiple visual themes for different aesthetics  
  • Orbital mechanics with Earth-Moon system
  • Educational labels with real physics data
  • High-quality exports for presentations
  
📚 PHYSICS: This visualization demonstrates Einstein's General 
Relativity, showing how mass curves spacetime according to 
the field equation: Gμν = 8πTμν
        """
        print(help_text)
    
    # Connect event handler
    fig.canvas.mpl_connect('key_press_event', on_key_advanced)
    
    # Show initial help
    print("\n" + "="*60)
    print("🌌 ENHANCED SPACETIME CURVATURE VISUALIZER")
    print("="*60)
    print("Press 'H' for full help menu")
    print("Press 'A' to start orbital animation")
    print("Press 'M' to cycle visual modes")
    print("Use mouse to rotate, arrow keys for fine control")
    print("="*60)

if __name__ == "__main__":
    print("🚀 Initializing Enhanced Spacetime Curvature Visualizer...")
    
    # Set matplotlib to use interactive backend
    plt.ion()
    
    # Create enhanced visualization
    fig, ax = create_enhanced_spacetime_visualization()
    
    # Add advanced interactive controls
    add_advanced_controls()
    
    # Show the visualization
    plt.show(block=True)
    
    print("\n🌌 Visualization session ended.")
    print("Thank you for exploring spacetime curvature!")

