import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns

# Set up the plotting style
plt.style.use('dark_background')
sns.set_palette("husl")

class TimeDilationAnalysis:
    def __init__(self):
        self.c = 1.0  # Speed of light (normalized)
        
    def lorentz_factor(self, velocity):
        """Calculate the Lorentz factor (gamma)"""
        return 1.0 / np.sqrt(1 - (velocity/self.c)**2)
    
    def time_dilation_factor(self, velocity):
        """Calculate time dilation factor"""
        return 1.0 / self.lorentz_factor(velocity)
    
    def plot_lorentz_factor(self):
        """Plot Lorentz factor vs velocity"""
        velocities = np.linspace(0.01, 0.999, 1000)
        gamma_values = [self.lorentz_factor(v) for v in velocities]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear scale
        ax1.plot(velocities, gamma_values, 'cyan', linewidth=2)
        ax1.set_xlabel('Velocity (fraction of c)')
        ax1.set_ylabel('Lorentz Factor (γ)')
        ax1.set_title('Lorentz Factor vs Velocity (Linear Scale)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 20)
        
        # Log scale
        ax2.semilogy(velocities, gamma_values, 'orange', linewidth=2)
        ax2.set_xlabel('Velocity (fraction of c)')
        ax2.set_ylabel('Lorentz Factor (γ) - Log Scale')
        ax2.set_title('Lorentz Factor vs Velocity (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        # Add significant velocity markers
        significant_velocities = [0.1, 0.5, 0.9, 0.95, 0.99, 0.999]
        for v in significant_velocities:
            gamma = self.lorentz_factor(v)
            ax1.plot(v, gamma, 'ro', markersize=8)
            ax1.annotate(f'v={v}c\nγ={gamma:.2f}', 
                        xy=(v, gamma), xytext=(10, 10), 
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_time_comparison(self):
        """Plot time experienced by different observers"""
        coordinate_times = np.linspace(0, 10, 100)
        velocities = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot coordinate time (45-degree line)
        ax.plot(coordinate_times, coordinate_times, 'white', linewidth=2, 
                linestyle='--', label='Coordinate Time (Reference Frame)')
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(velocities)))
        
        for i, v in enumerate(velocities):
            proper_times = coordinate_times * self.time_dilation_factor(v)
            ax.plot(coordinate_times, proper_times, color=colors[i], 
                   linewidth=2, label=f'v = {v}c (γ = {self.lorentz_factor(v):.2f})')
        
        ax.set_xlabel('Coordinate Time (Reference Frame) [s]')
        ax.set_ylabel('Proper Time (Moving Observer) [s]')
        ax.set_title('Time Dilation: Proper Time vs Coordinate Time')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_animated_simulation(self):
        """Create an animated visualization of time dilation"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Animation parameters
        max_time = 10
        dt = 0.1
        times = np.arange(0, max_time, dt)
        
        # Ships data
        ship_a_velocity = 0.1
        ship_b_velocity = 0.9
        
        ship_a_times = []
        ship_b_times = []
        coord_times = []
        
        def animate(frame):
            coord_time = frame * dt
            ship_a_proper = coord_time * self.time_dilation_factor(ship_a_velocity)
            ship_b_proper = coord_time * self.time_dilation_factor(ship_b_velocity)
            
            coord_times.append(coord_time)
            ship_a_times.append(ship_a_proper)
            ship_b_times.append(ship_b_proper)
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Top plot: Visual representation
            ax1.set_xlim(0, 12)
            ax1.set_ylim(0, 8)
            ax1.set_title(f'Time Dilation Visualization (t = {coord_time:.1f}s)')
            
            # Draw spaceships
            ship_a_x = 1 + (coord_time * ship_a_velocity * 0.5) % 10
            ship_b_x = 1 + (coord_time * ship_b_velocity * 0.5) % 10
            
            # Ship A
            ship_a_rect = Rectangle((ship_a_x, 2), 1, 0.5, facecolor='cyan', alpha=0.7)
            ax1.add_patch(ship_a_rect)
            ax1.text(ship_a_x, 3, f'Ship A\nv={ship_a_velocity}c\nClock: {ship_a_proper:.2f}s', 
                    ha='center', fontsize=10, color='cyan')
            
            # Ship B
            ship_b_rect = Rectangle((ship_b_x, 5), 1, 0.5, facecolor='orange', alpha=0.7)
            ax1.add_patch(ship_b_rect)
            ax1.text(ship_b_x, 6, f'Ship B\nv={ship_b_velocity}c\nClock: {ship_b_proper:.2f}s', 
                    ha='center', fontsize=10, color='orange')
            
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Ships')
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Time comparison graph
            if len(coord_times) > 1:
                ax2.plot(coord_times, coord_times, 'white', linestyle='--', 
                        linewidth=2, label='Coordinate Time')
                ax2.plot(coord_times, ship_a_times, 'cyan', linewidth=2, 
                        label=f'Ship A (v={ship_a_velocity}c)')
                ax2.plot(coord_times, ship_b_times, 'orange', linewidth=2, 
                        label=f'Ship B (v={ship_b_velocity}c)')
            
            ax2.set_xlabel('Coordinate Time [s]')
            ax2.set_ylabel('Proper Time [s]')
            ax2.set_title('Clock Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, max_time)
            ax2.set_ylim(0, max_time)
            
            # Add time difference annotation
            time_diff = abs(ship_a_proper - ship_b_proper)
            ax2.text(0.7 * max_time, 0.9 * max_time, 
                    f'Time Difference: {time_diff:.2f}s\nγ_A = {self.lorentz_factor(ship_a_velocity):.2f}\nγ_B = {self.lorentz_factor(ship_b_velocity):.2f}',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    fontsize=12)
        
        anim = animation.FuncAnimation(fig, animate, frames=int(max_time/dt), 
                                     interval=100, repeat=True, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def calculate_twin_paradox(self, travel_velocity=0.9, travel_time_coordinate=10):
        """Calculate the famous twin paradox scenario"""
        gamma = self.lorentz_factor(travel_velocity)
        
        # Time experienced by traveling twin
        travel_time_proper = travel_time_coordinate / gamma
        
        # Time experienced by Earth twin
        earth_time = travel_time_coordinate
        
        age_difference = earth_time - travel_time_proper
        
        print("="*60)
        print("TWIN PARADOX CALCULATION")
        print("="*60)
        print(f"Travel velocity: {travel_velocity}c")
        print(f"Lorentz factor (γ): {gamma:.3f}")
        print(f"Trip duration (coordinate time): {travel_time_coordinate} years")
        print(f"Time experienced by traveling twin: {travel_time_proper:.3f} years")
        print(f"Time experienced by Earth twin: {earth_time} years")
        print(f"Age difference upon return: {age_difference:.3f} years")
        print(f"The traveling twin ages {age_difference:.3f} years less!")
        print("="*60)
        
        return {
            'travel_velocity': travel_velocity,
            'gamma': gamma,
            'coordinate_time': travel_time_coordinate,
            'traveler_time': travel_time_proper,
            'earth_time': earth_time,
            'age_difference': age_difference
        }
    
    def plot_real_world_examples(self):
        """Plot time dilation effects for real-world scenarios"""
        # Real-world examples with actual velocities
        examples = {
            'Commercial Jet': 250,  # m/s
            'ISS': 7660,  # m/s
            'Voyager 1': 17000,  # m/s
            'Parker Solar Probe': 200000,  # m/s (at closest approach)
            'LHC Protons': 299792455,  # m/s (99.9999991% c)
            'Cosmic Ray Protons': 299792457  # m/s (99.99999999% c)
        }
        
        c_actual = 299792458  # m/s
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        names = []
        velocities_fraction = []
        gamma_factors = []
        time_dilations = []
        
        for name, velocity in examples.items():
            v_fraction = velocity / c_actual
            gamma = self.lorentz_factor(v_fraction)
            time_dilation = (gamma - 1) * 100  # Percentage difference
            
            names.append(name)
            velocities_fraction.append(v_fraction)
            gamma_factors.append(gamma)
            time_dilations.append(time_dilation)
        
        # Velocity comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars1 = ax1.barh(names, velocities_fraction, color=colors)
        ax1.set_xlabel('Velocity (fraction of c)')
        ax1.set_title('Real-World Velocities')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Add velocity labels
        for i, (bar, v) in enumerate(zip(bars1, velocities_fraction)):
            ax1.text(v * 1.1, bar.get_y() + bar.get_height()/2, 
                    f'{v:.2e}c', va='center', fontsize=8)
        
        # Time dilation effects
        bars2 = ax2.barh(names, time_dilations, color=colors)
        ax2.set_xlabel('Time Dilation Effect (%)')
        ax2.set_title('Time Dilation Effects')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (bar, td) in enumerate(zip(bars2, time_dilations)):
            if td > 1e-10:  # Only show meaningful values
                ax2.text(td * 1.1, bar.get_y() + bar.get_height()/2, 
                        f'{td:.2e}%', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed information
        print("\nREAL-WORLD TIME DILATION EFFECTS")
        print("="*60)
        for name, v_frac, gamma, td in zip(names, velocities_fraction, gamma_factors, time_dilations):
            print(f"{name:20s}: v={v_frac:.2e}c, γ={gamma:.10f}, Δt={td:.2e}%")
        
        return fig
    
    def comprehensive_analysis(self):
        """Run a comprehensive analysis of time dilation effects"""
        print("🚀 COMPREHENSIVE TIME DILATION ANALYSIS 🚀")
        print("="*60)
        
        # Generate all plots
        print("Generating Lorentz factor plot...")
        self.plot_lorentz_factor()
        
        print("Generating time comparison plot...")
        self.plot_time_comparison()
        
        print("Generating real-world examples plot...")
        self.plot_real_world_examples()
        
        # Calculate twin paradox scenarios
        print("\nCalculating twin paradox scenarios...")
        scenarios = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
        for v in scenarios:
            self.calculate_twin_paradox(v, 10)
            print()
        
        print("Analysis complete! 🎉")
        
        # Create animation (optional - comment out if too resource intensive)
        print("Creating animated simulation...")
        try:
            anim = self.create_animated_simulation()
            print("Animation created successfully!")
            return anim
        except Exception as e:
            print(f"Animation creation failed: {e}")
            return None

if __name__ == "__main__":
    analyzer = TimeDilationAnalysis()
    analyzer.comprehensive_analysis() 