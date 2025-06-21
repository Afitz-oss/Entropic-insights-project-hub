import pygame
import math
import sys
from typing import Tuple, List
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
SPEED_OF_LIGHT = 1.0  # Normalized units (c = 1)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (100, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

class JourneyPhase(Enum):
    OUTBOUND = "Outbound Journey"
    TURNAROUND = "Turnaround"
    INBOUND = "Return Journey"
    REUNION = "Reunion Complete"

class Spaceship:
    def __init__(self, name: str, color: Tuple[int, int, int], initial_velocity: float, y_position: int, crew_name: str, initial_age: float):
        self.name = name
        self.color = color
        self.cruise_velocity = initial_velocity  # Cruise velocity for the journey
        self.velocity = 0.0 if name == "Earth Station" else initial_velocity  # Current velocity
        self.proper_time = 0.0  # Time experienced by the spaceship (in years)
        self.coordinate_time = 0.0  # Time in the reference frame (in years)
        self.y_position = y_position
        self.x_position = 100
        self.crew_name = crew_name
        self.initial_age = initial_age
        self.current_age = initial_age
        
        # Journey tracking for traveling ship
        self.is_traveling = name != "Earth Station"
        self.journey_phase = JourneyPhase.OUTBOUND if self.is_traveling else None
        self.distance_traveled = 0.0  # Distance from Earth (in light-years)
        self.max_distance = 10.0  # Maximum distance before turnaround (light-years)
        self.turnaround_time = 0.5  # Time spent in turnaround phase (years)
        self.turnaround_elapsed = 0.0
        self.has_returned = False
        
    def update_time(self, dt: float):
        """Update time based on special relativity and journey phase"""
        # Update coordinate time (reference frame time)
        self.coordinate_time += dt
        
        # Handle journey phases for traveling spaceship
        if self.is_traveling and not self.has_returned:
            self._update_journey_phase(dt)
        
        # Calculate proper time based on current velocity
        if self.velocity == 0.0:
            # Stationary observer experiences coordinate time normally
            self.proper_time = self.coordinate_time
        else:
            # Calculate Lorentz factor (gamma)
            gamma = 1.0 / math.sqrt(1 - (self.velocity ** 2))
            # From the reference frame: proper_time = coordinate_time / gamma
            self.proper_time = self.coordinate_time / gamma
        
        # Update current age
        self.current_age = self.initial_age + self.proper_time
        
        # Update visual position
        self._update_position(dt)
    
    def _update_journey_phase(self, dt: float):
        """Update the journey phase and distance for traveling spaceship"""
        if self.journey_phase == JourneyPhase.OUTBOUND:
            # Travel away from Earth
            self.distance_traveled += self.velocity * dt
            if self.distance_traveled >= self.max_distance:
                self.journey_phase = JourneyPhase.TURNAROUND
                self.turnaround_elapsed = 0.0
                
        elif self.journey_phase == JourneyPhase.TURNAROUND:
            # Turnaround phase - ship is accelerating/decelerating
            self.turnaround_elapsed += dt
            if self.turnaround_elapsed >= self.turnaround_time:
                self.journey_phase = JourneyPhase.INBOUND
                self.velocity = self.cruise_velocity  # Resume cruise velocity for return
                
        elif self.journey_phase == JourneyPhase.INBOUND:
            # Return to Earth
            self.distance_traveled -= self.velocity * dt
            if self.distance_traveled <= 0:
                self.distance_traveled = 0
                self.velocity = 0
                self.journey_phase = JourneyPhase.REUNION
                self.has_returned = True
    
    def _update_position(self, dt: float):
        """Update visual position based on journey phase"""
        if not self.is_traveling:
            return
            
        if self.journey_phase == JourneyPhase.OUTBOUND:
            # Move right (away from Earth)
            self.x_position += 30 * dt
            if self.x_position > WINDOW_WIDTH - 100:
                self.x_position = WINDOW_WIDTH - 100
                
        elif self.journey_phase == JourneyPhase.TURNAROUND:
            # Stay at maximum distance during turnaround
            self.x_position = WINDOW_WIDTH - 100
            
        elif self.journey_phase == JourneyPhase.INBOUND:
            # Move left (back to Earth)
            self.x_position -= 30 * dt
            if self.x_position < 100:
                self.x_position = 100
                
        elif self.journey_phase == JourneyPhase.REUNION:
            # Stay at Earth
            self.x_position = 100
    
    def get_gamma(self) -> float:
        """Calculate and return the Lorentz factor"""
        if self.velocity == 0.0:
            return 1.0
        return 1.0 / math.sqrt(1 - (self.velocity ** 2))
    
    def get_journey_progress(self) -> float:
        """Get journey progress as a percentage (0-100)"""
        if not self.is_traveling:
            return 0
        
        total_distance = self.max_distance * 2  # Round trip
        
        if self.journey_phase == JourneyPhase.OUTBOUND:
            return (self.distance_traveled / total_distance) * 100
        elif self.journey_phase == JourneyPhase.TURNAROUND:
            outbound_progress = self.max_distance / total_distance
            turnaround_progress = (self.turnaround_elapsed / self.turnaround_time) * 0.05  # Small addition for turnaround
            return (outbound_progress + turnaround_progress) * 100
        elif self.journey_phase == JourneyPhase.INBOUND:
            inbound_distance = self.max_distance - self.distance_traveled
            return ((self.max_distance + inbound_distance) / total_distance) * 100
        else:  # REUNION
            return 100

class TimeDialationSimulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Twin Paradox: Complete Round-Trip Journey")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_huge = pygame.font.Font(None, 48)
        
        # Configuration options
        self.config_velocity = 0.8  # Bob's cruise velocity
        self.config_distance = 10.0  # Maximum journey distance
        self.config_time_speed = 1.0  # Time acceleration
        
        # Create spaceships - Twin Paradox scenario
        self.ship_a = Spaceship("Earth Station", BLUE, 0.0, 200, "Alice", 25.0)  # Stays on Earth
        self.ship_b = Spaceship("Deep Space Mission", RED, self.config_velocity, 400, "Bob", 25.0)  # Travels at high speed
        
        self.time_acceleration = self.config_time_speed
        self.paused = False
        self.show_graph = True
        self.time_history = []  # Store time data for graphing
        self.reunion_detected = False
        self.simulation_complete = False
        self.show_config_screen = False
        
        # Final results storage
        self.final_results = {}
        
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if self.show_config_screen:
                    return self.handle_config_events(event)
                elif self.simulation_complete:
                    return self.handle_completion_events(event)
                else:
                    return self.handle_simulation_events(event)
        return True
    
    def handle_config_events(self, event):
        """Handle events in configuration screen"""
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_c:
            self.show_config_screen = False
        elif event.key == pygame.K_RETURN or event.key == pygame.K_s:
            self.start_new_simulation()
            self.show_config_screen = False
        elif event.key == pygame.K_UP:
            self.config_velocity = min(0.995, self.config_velocity + 0.05)
        elif event.key == pygame.K_DOWN:
            self.config_velocity = max(0.1, self.config_velocity - 0.05)
        elif event.key == pygame.K_RIGHT:
            self.config_distance = min(25.0, self.config_distance + 1.0)
        elif event.key == pygame.K_LEFT:
            self.config_distance = max(2.0, self.config_distance - 1.0)
        elif event.key == pygame.K_PAGEUP:
            self.config_time_speed = min(10.0, self.config_time_speed + 0.5)
        elif event.key == pygame.K_PAGEDOWN:
            self.config_time_speed = max(0.1, self.config_time_speed - 0.5)
        elif event.key == pygame.K_1:
            self.load_preset_config("fast")
        elif event.key == pygame.K_2:
            self.load_preset_config("moderate")
        elif event.key == pygame.K_3:
            self.load_preset_config("extreme")
        return True
    
    def handle_completion_events(self, event):
        """Handle events in completion screen"""
        if event.key == pygame.K_r:
            self.reset_simulation()
        elif event.key == pygame.K_c:
            self.show_config_screen = True
            self.simulation_complete = False
        elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
            return False
        return True
    
    def handle_simulation_events(self, event):
        """Handle events during simulation"""
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_r:
            self.reset_simulation()
        elif event.key == pygame.K_c:
            self.show_config_screen = True
        elif event.key == pygame.K_g:
            self.show_graph = not self.show_graph
        elif event.key == pygame.K_UP:
            self.time_acceleration = min(5.0, self.time_acceleration + 0.2)
        elif event.key == pygame.K_DOWN:
            self.time_acceleration = max(0.1, self.time_acceleration - 0.2)
        elif event.key == pygame.K_LEFT:
            if not self.ship_b.has_returned:
                self.ship_b.cruise_velocity = max(0.1, self.ship_b.cruise_velocity - 0.05)
                if self.ship_b.journey_phase in [JourneyPhase.OUTBOUND, JourneyPhase.INBOUND]:
                    self.ship_b.velocity = self.ship_b.cruise_velocity
        elif event.key == pygame.K_RIGHT:
            if not self.ship_b.has_returned:
                self.ship_b.cruise_velocity = min(0.99, self.ship_b.cruise_velocity + 0.05)
                if self.ship_b.journey_phase in [JourneyPhase.OUTBOUND, JourneyPhase.INBOUND]:
                    self.ship_b.velocity = self.ship_b.cruise_velocity
        elif event.key == pygame.K_d:
            # Adjust journey distance
            if not self.ship_b.has_returned:
                self.ship_b.max_distance = min(20.0, self.ship_b.max_distance + 1.0)
        elif event.key == pygame.K_s:
            # Adjust journey distance
            if not self.ship_b.has_returned:
                self.ship_b.max_distance = max(2.0, self.ship_b.max_distance - 1.0)
        return True
    
    def load_preset_config(self, preset_name):
        """Load preset configurations"""
        presets = {
            "fast": {"velocity": 0.6, "distance": 5.0, "time_speed": 2.0},
            "moderate": {"velocity": 0.8, "distance": 10.0, "time_speed": 1.0},
            "extreme": {"velocity": 0.95, "distance": 20.0, "time_speed": 0.5}
        }
        if preset_name in presets:
            preset = presets[preset_name]
            self.config_velocity = preset["velocity"]
            self.config_distance = preset["distance"]
            self.config_time_speed = preset["time_speed"]
    
    def start_new_simulation(self):
        """Start a new simulation with current configuration"""
        self.ship_a = Spaceship("Earth Station", BLUE, 0.0, 200, "Alice", 25.0)
        self.ship_b = Spaceship("Deep Space Mission", RED, self.config_velocity, 400, "Bob", 25.0)
        self.ship_b.max_distance = self.config_distance
        
        self.time_acceleration = self.config_time_speed
        self.paused = False
        self.time_history.clear()
        self.reunion_detected = False
        self.simulation_complete = False
        self.final_results = {}
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.ship_a.proper_time = 0.0
        self.ship_a.coordinate_time = 0.0
        self.ship_a.current_age = self.ship_a.initial_age
        self.ship_a.x_position = 100
        
        self.ship_b.proper_time = 0.0
        self.ship_b.coordinate_time = 0.0
        self.ship_b.current_age = self.ship_b.initial_age
        self.ship_b.x_position = 100
        self.ship_b.velocity = self.ship_b.cruise_velocity
        self.ship_b.journey_phase = JourneyPhase.OUTBOUND
        self.ship_b.distance_traveled = 0.0
        self.ship_b.turnaround_elapsed = 0.0
        self.ship_b.has_returned = False
        
        self.time_history.clear()
        self.reunion_detected = False
        self.simulation_complete = False
        self.final_results = {}
    
    def update(self, dt: float):
        """Update simulation state"""
        if not self.paused and not self.simulation_complete:
            adjusted_dt = dt * self.time_acceleration
            self.ship_a.update_time(adjusted_dt)
            self.ship_b.update_time(adjusted_dt)
            
            # Check for reunion and end simulation
            if self.ship_b.has_returned and not self.reunion_detected:
                self.reunion_detected = True
                self.simulation_complete = True
                self.calculate_final_results()
                self.paused = True  # Auto-pause when complete
            
            # Store data for graphing
            if len(self.time_history) < 2000:  # Increased history size for longer journey
                self.time_history.append({
                    'coord_time': self.ship_a.coordinate_time,
                    'ship_a_proper': self.ship_a.proper_time,
                    'ship_b_proper': self.ship_b.proper_time,
                    'distance': self.ship_b.distance_traveled,
                    'phase': self.ship_b.journey_phase.value if self.ship_b.journey_phase else "Stationary"
                })
    
    def calculate_final_results(self):
        """Calculate and store final results"""
        self.final_results = {
            'journey_duration_earth': self.ship_a.coordinate_time,
            'alice_aged': self.ship_a.proper_time,
            'bob_aged': self.ship_b.proper_time,
            'age_difference': self.ship_a.current_age - self.ship_b.current_age,
            'time_dilation_factor': self.ship_b.get_gamma(),
            'journey_distance': self.ship_b.max_distance,
            'cruise_velocity': self.ship_b.cruise_velocity,
            'theoretical_slowdown': 1.0 / self.ship_b.get_gamma()
        }
    
    def draw_spaceship(self, ship: Spaceship):
        """Draw a spaceship with its crew information"""
        # Draw spaceship body with phase-specific styling
        ship_rect = pygame.Rect(ship.x_position, ship.y_position, 80, 30)
        
        # Color based on journey phase
        if ship.is_traveling and ship.journey_phase:
            if ship.journey_phase == JourneyPhase.TURNAROUND:
                border_color = ORANGE
            elif ship.journey_phase == JourneyPhase.REUNION:
                border_color = GREEN
            else:
                border_color = WHITE
        else:
            border_color = WHITE
            
        pygame.draw.rect(self.screen, ship.color, ship_rect)
        pygame.draw.rect(self.screen, border_color, ship_rect, 3)
        
        # Draw ship name
        name_text = self.font_medium.render(ship.name, True, WHITE)
        self.screen.blit(name_text, (ship.x_position, ship.y_position - 70))
        
        # Draw journey phase for traveling ship
        if ship.is_traveling and ship.journey_phase:
            phase_text = self.font_small.render(f"Phase: {ship.journey_phase.value}", True, YELLOW)
            self.screen.blit(phase_text, (ship.x_position, ship.y_position - 50))
        
        # Draw crew name
        crew_text = self.font_small.render(f"Crew: {ship.crew_name}", True, WHITE)
        self.screen.blit(crew_text, (ship.x_position, ship.y_position - 30))
        
        # Draw velocity indicator
        velocity_text = f"v = {ship.velocity:.2f}c"
        vel_surface = self.font_small.render(velocity_text, True, WHITE)
        self.screen.blit(vel_surface, (ship.x_position, ship.y_position + 35))
        
        # Draw current age
        age_text = f"Age: {ship.current_age:.1f} years"
        age_surface = self.font_medium.render(age_text, True, ship.color)
        self.screen.blit(age_surface, (ship.x_position + 100, ship.y_position - 10))
        
        # Draw elapsed time
        elapsed_text = f"Experienced: {ship.proper_time:.1f} years"
        elapsed_surface = self.font_small.render(elapsed_text, True, ship.color)
        self.screen.blit(elapsed_surface, (ship.x_position + 100, ship.y_position + 10))
        
        # Draw gamma factor
        gamma_text = f"γ = {ship.get_gamma():.2f}"
        gamma_surface = self.font_small.render(gamma_text, True, YELLOW)
        self.screen.blit(gamma_surface, (ship.x_position + 100, ship.y_position + 30))
        
        # Draw journey progress for traveling ship
        if ship.is_traveling:
            progress = ship.get_journey_progress()
            progress_text = f"Journey: {progress:.1f}%"
            progress_surface = self.font_small.render(progress_text, True, ORANGE)
            self.screen.blit(progress_surface, (ship.x_position + 100, ship.y_position + 50))
    
    def draw_journey_timeline(self):
        """Draw a visual timeline of the journey"""
        timeline_x = 20
        timeline_y = 120
        timeline_width = 300
        timeline_height = 20
        
        # Draw timeline background
        timeline_rect = pygame.Rect(timeline_x, timeline_y, timeline_width, timeline_height)
        pygame.draw.rect(self.screen, GRAY, timeline_rect)
        pygame.draw.rect(self.screen, WHITE, timeline_rect, 2)
        
        # Draw progress
        progress = self.ship_b.get_journey_progress()
        progress_width = int((progress / 100) * timeline_width)
        if progress_width > 0:
            progress_rect = pygame.Rect(timeline_x, timeline_y, progress_width, timeline_height)
            
            # Color based on phase
            if self.ship_b.journey_phase == JourneyPhase.OUTBOUND:
                color = RED
            elif self.ship_b.journey_phase == JourneyPhase.TURNAROUND:
                color = ORANGE
            elif self.ship_b.journey_phase == JourneyPhase.INBOUND:
                color = BLUE
            else:
                color = GREEN
                
            pygame.draw.rect(self.screen, color, progress_rect)
        
        # Draw markers
        half_point = timeline_x + timeline_width // 2
        pygame.draw.line(self.screen, WHITE, (half_point, timeline_y), (half_point, timeline_y + timeline_height), 2)
        
        # Labels
        title = self.font_small.render("Journey Timeline", True, WHITE)
        self.screen.blit(title, (timeline_x, timeline_y - 20))
        
        earth_label = self.font_small.render("Earth", True, WHITE)
        self.screen.blit(earth_label, (timeline_x, timeline_y + timeline_height + 5))
        
        turnaround_label = self.font_small.render("Turnaround", True, WHITE)
        self.screen.blit(turnaround_label, (half_point - 30, timeline_y + timeline_height + 5))
        
        return_label = self.font_small.render("Return", True, WHITE)
        self.screen.blit(return_label, (timeline_x + timeline_width - 30, timeline_y + timeline_height + 5))
    
    def draw_graph(self):
        """Draw time comparison graph"""
        if not self.time_history or not self.show_graph:
            return
            
        graph_x = WINDOW_WIDTH - 450
        graph_y = 50
        graph_width = 400
        graph_height = 250
        
        # Draw graph background
        graph_rect = pygame.Rect(graph_x, graph_y, graph_width, graph_height)
        pygame.draw.rect(self.screen, GRAY, graph_rect)
        pygame.draw.rect(self.screen, WHITE, graph_rect, 2)
        
        # Graph title
        title = self.font_medium.render("Age vs Journey Time", True, WHITE)
        self.screen.blit(title, (graph_x + 10, graph_y - 25))
        
        # Draw axes
        pygame.draw.line(self.screen, WHITE, (graph_x, graph_y + graph_height), 
                        (graph_x + graph_width, graph_y + graph_height), 2)
        pygame.draw.line(self.screen, WHITE, (graph_x, graph_y), 
                        (graph_x, graph_y + graph_height), 2)
        
        # Add axis labels
        x_label = self.font_small.render("Coordinate Time (years)", True, WHITE)
        self.screen.blit(x_label, (graph_x + graph_width//2 - 60, graph_y + graph_height + 10))
        
        # Plot data points
        if len(self.time_history) > 1:
            max_coord_time = max(entry['coord_time'] for entry in self.time_history)
            max_age = max(self.ship_a.current_age, self.ship_b.current_age)
            
            if max_coord_time > 0 and max_age > 0:
                prev_x = None
                prev_y_a = None
                prev_y_b = None
                
                for entry in self.time_history:
                    # Scale coordinates
                    x = graph_x + (entry['coord_time'] / max_coord_time) * graph_width
                    
                    # Alice's age (blue)
                    alice_age = self.ship_a.initial_age + entry['ship_a_proper']
                    y_a = graph_y + graph_height - ((alice_age - self.ship_a.initial_age) / (max_age - self.ship_a.initial_age)) * graph_height
                    
                    # Bob's age (red)
                    bob_age = self.ship_b.initial_age + entry['ship_b_proper']
                    y_b = graph_y + graph_height - ((bob_age - self.ship_b.initial_age) / (max_age - self.ship_a.initial_age)) * graph_height
                    
                    if prev_x is not None:
                        pygame.draw.line(self.screen, BLUE, (prev_x, prev_y_a), (x, y_a), 2)
                        pygame.draw.line(self.screen, RED, (prev_x, prev_y_b), (x, y_b), 2)
                    
                    prev_x = x
                    prev_y_a = y_a
                    prev_y_b = y_b
                
                # Draw current ages as dots
                if max_coord_time > 0:
                    current_x = graph_x + (self.ship_a.coordinate_time / max_coord_time) * graph_width
                    alice_current_y = graph_y + graph_height - ((self.ship_a.current_age - self.ship_a.initial_age) / (max_age - self.ship_a.initial_age)) * graph_height
                    bob_current_y = graph_y + graph_height - ((self.ship_b.current_age - self.ship_b.initial_age) / (max_age - self.ship_a.initial_age)) * graph_height
                    
                    pygame.draw.circle(self.screen, BLUE, (int(current_x), int(alice_current_y)), 4)
                    pygame.draw.circle(self.screen, RED, (int(current_x), int(bob_current_y)), 4)
        
        # Graph legend
        legend_y = graph_y + graph_height + 30
        pygame.draw.line(self.screen, BLUE, (graph_x, legend_y), (graph_x + 20, legend_y), 3)
        blue_label = self.font_small.render("Alice (Earth)", True, WHITE)
        self.screen.blit(blue_label, (graph_x + 25, legend_y - 8))
        
        pygame.draw.line(self.screen, RED, (graph_x + 120, legend_y), (graph_x + 140, legend_y), 3)
        red_label = self.font_small.render("Bob (Space)", True, WHITE)
        self.screen.blit(red_label, (graph_x + 145, legend_y - 8))
        
        # Show age difference if journey is complete
        if self.ship_b.has_returned:
            age_diff = self.ship_a.current_age - self.ship_b.current_age
            diff_text = f"Final Age Difference: {age_diff:.1f} years"
            diff_surface = self.font_medium.render(diff_text, True, GREEN)
            self.screen.blit(diff_surface, (graph_x, legend_y + 20))
    
    def draw_info_panel(self):
        """Draw information panel with physics explanations and controls"""
        info_x = 20
        info_y = 500
        
        # Physics explanation with journey context
        if self.ship_b.has_returned:
            age_difference = self.ship_a.current_age - self.ship_b.current_age
            journey_time = self.ship_a.coordinate_time
            explanation = [
                "🎉 REUNION COMPLETE! 🎉",
                f"Journey Duration: {journey_time:.1f} years (Earth time)",
                f"Alice aged: {self.ship_a.proper_time:.1f} years",
                f"Bob aged: {self.ship_b.proper_time:.1f} years",
                f"Age difference: {abs(age_difference):.1f} years",
                "Bob is now younger than Alice!"
            ]
        else:
            explanation = [
                "TWIN PARADOX SIMULATION",
                "Round-trip journey with time dilation",
                f"Current phase: {self.ship_b.journey_phase.value}",
                f"Distance from Earth: {self.ship_b.distance_traveled:.1f} ly",
                f"Bob's velocity: {self.ship_b.velocity:.2f}c",
                f"Time dilation factor: γ = {self.ship_b.get_gamma():.2f}"
            ]
        
        for i, text in enumerate(explanation):
            color = GREEN if "REUNION" in text else WHITE
            if "🎉" in text:
                color = YELLOW
            surface = self.font_medium.render(text, True, color)
            self.screen.blit(surface, (info_x, info_y + i * 25))
        
        # Controls
        controls_y = info_y + 170
        if self.simulation_complete:
            controls = [
                "Controls:",
                "R - Restart Same Journey  |  C - Configure New Journey",
                "Q/ESC - Quit Application"
            ]
        else:
            controls = [
                "Controls:",
                "SPACE - Pause/Resume  |  R - Reset  |  C - Configure",
                "G - Toggle graph      |  ↑/↓ - Time speed",
                "←/→ - Bob's velocity  |  D/S - Journey distance"
            ]
        
        for i, text in enumerate(controls):
            color = YELLOW if i == 0 else WHITE
            surface = self.font_small.render(text, True, color)
            self.screen.blit(surface, (info_x, controls_y + i * 20))
        
        # Current status info
        status_x = info_x + 400
        status_y = info_y
        
        total_journey_time = (self.ship_b.max_distance * 2) / self.ship_b.cruise_velocity + self.ship_b.turnaround_time
        eta_text = f"Est. journey time: {total_journey_time:.1f} years" if not self.ship_b.has_returned else "Journey complete!"
        
        status_info = [
            f"Time Speed: {self.time_acceleration:.1f}x",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}",
            f"Journey Distance: {self.ship_b.max_distance:.1f} light-years",
            eta_text,
            f"Progress: {self.ship_b.get_journey_progress():.1f}%"
        ]
        
        for i, text in enumerate(status_info):
            color = GREEN if "complete" in text else WHITE
            surface = self.font_small.render(text, True, color)
            self.screen.blit(surface, (status_x, status_y + i * 25))
        
        # Show reunion message if just completed
        if self.reunion_detected and self.ship_b.has_returned:
            reunion_text = "Welcome back, Bob! The twins are reunited."
            reunion_surface = self.font_large.render(reunion_text, True, GREEN)
            reunion_rect = reunion_surface.get_rect(center=(WINDOW_WIDTH // 2, 350))
            
            # Draw background for better visibility
            bg_rect = reunion_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, BLACK, bg_rect)
            pygame.draw.rect(self.screen, GREEN, bg_rect, 2)
            
            self.screen.blit(reunion_surface, reunion_rect)
    
    def draw_config_screen(self):
        """Draw configuration screen"""
        self.screen.fill(BLACK)
        
        # Title
        title = self.font_huge.render("Twin Paradox Configuration", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)
        
        # Configuration options
        config_y = 180
        line_height = 40
        
        # Bob's velocity
        velocity_text = f"Bob's Cruise Velocity: {self.config_velocity:.2f}c"
        velocity_color = RED if self.config_velocity > 0.9 else YELLOW if self.config_velocity > 0.7 else WHITE
        velocity_surface = self.font_large.render(velocity_text, True, velocity_color)
        self.screen.blit(velocity_surface, (100, config_y))
        
        # Journey distance
        distance_text = f"Journey Distance: {self.config_distance:.1f} light-years"
        distance_surface = self.font_large.render(distance_text, True, WHITE)
        self.screen.blit(distance_surface, (100, config_y + line_height))
        
        # Time speed
        time_speed_text = f"Time Acceleration: {self.config_time_speed:.1f}x"
        time_speed_surface = self.font_large.render(time_speed_text, True, WHITE)
        self.screen.blit(time_speed_surface, (100, config_y + line_height * 2))
        
        # Calculated journey time
        theoretical_time = (self.config_distance * 2) / self.config_velocity + 0.5
        gamma = 1.0 / math.sqrt(1 - (self.config_velocity ** 2))
        bob_experience = theoretical_time / gamma
        
        calc_y = config_y + line_height * 4
        journey_info = [
            f"Estimated Journey Time (Earth): {theoretical_time:.1f} years",
            f"Time Dilation Factor (γ): {gamma:.2f}",
            f"Bob Will Experience: {bob_experience:.1f} years",
            f"Age Difference at Return: ~{theoretical_time - bob_experience:.1f} years"
        ]
        
        for i, info in enumerate(journey_info):
            info_surface = self.font_medium.render(info, True, LIGHT_GRAY)
            self.screen.blit(info_surface, (100, calc_y + i * 25))
        
        # Preset buttons
        presets_y = calc_y + 120
        presets_title = self.font_medium.render("Quick Presets:", True, YELLOW)
        self.screen.blit(presets_title, (100, presets_y))
        
        preset_info = [
            "1 - Fast Journey (0.6c, 5ly, 2x speed)",
            "2 - Moderate Journey (0.8c, 10ly, 1x speed)",
            "3 - Extreme Journey (0.95c, 20ly, 0.5x speed)"
        ]
        
        for i, preset in enumerate(preset_info):
            preset_surface = self.font_small.render(preset, True, WHITE)
            self.screen.blit(preset_surface, (120, presets_y + 25 + i * 20))
        
        # Controls
        controls_y = presets_y + 110
        controls = [
            "Controls:",
            "↑/↓ - Adjust velocity",
            "←/→ - Adjust distance", 
            "PgUp/PgDn - Adjust time speed",
            "ENTER/S - Start simulation",
            "ESC/C - Cancel"
        ]
        
        for i, control in enumerate(controls):
            color = YELLOW if i == 0 else WHITE
            control_surface = self.font_small.render(control, True, color)
            self.screen.blit(control_surface, (100, controls_y + i * 20))
        
        pygame.display.flip()
    
    def draw_completion_screen(self):
        """Draw completion screen with final results"""
        # Draw background simulation state (faded)
        self.draw_simulation()
        
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Main completion message
        completion_title = self.font_huge.render("🎉 JOURNEY COMPLETE! 🎉", True, GREEN)
        title_rect = completion_title.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(completion_title, title_rect)
        
        # Final results box
        results_x = WINDOW_WIDTH // 2 - 300
        results_y = 180
        results_width = 600
        results_height = 350
        
        results_rect = pygame.Rect(results_x, results_y, results_width, results_height)
        pygame.draw.rect(self.screen, GRAY, results_rect)
        pygame.draw.rect(self.screen, WHITE, results_rect, 3)
        
        # Results title
        results_title = self.font_large.render("Final Results", True, WHITE)
        title_rect = results_title.get_rect(center=(WINDOW_WIDTH // 2, results_y + 30))
        self.screen.blit(results_title, title_rect)
        
        # Results data
        if self.final_results:
            results_text_y = results_y + 70
            line_height = 30
            
            results_data = [
                f"Journey Duration (Earth Time): {self.final_results['journey_duration_earth']:.1f} years",
                f"Alice aged: {self.final_results['alice_aged']:.1f} years",
                f"Bob aged: {self.final_results['bob_aged']:.1f} years",
                f"Age Difference: {self.final_results['age_difference']:.1f} years",
                "",
                f"Journey Distance: {self.final_results['journey_distance']:.1f} light-years",
                f"Cruise Velocity: {self.final_results['cruise_velocity']:.2f}c",
                f"Time Dilation Factor: {self.final_results['time_dilation_factor']:.2f}",
                f"Bob experienced {self.final_results['theoretical_slowdown']:.1%} of Earth time"
            ]
            
            for i, result in enumerate(results_data):
                if result == "":
                    continue
                color = YELLOW if "Age Difference" in result else WHITE
                if "Bob experienced" in result:
                    color = GREEN
                result_surface = self.font_medium.render(result, True, color)
                result_rect = result_surface.get_rect(center=(WINDOW_WIDTH // 2, results_text_y + i * line_height))
                self.screen.blit(result_surface, result_rect)
        
        # Action buttons
        actions_y = results_y + results_height + 30
        actions = [
            "R - Run Same Configuration Again",
            "C - Configure New Journey",
            "Q/ESC - Quit"
        ]
        
        for i, action in enumerate(actions):
            action_surface = self.font_medium.render(action, True, YELLOW)
            action_rect = action_surface.get_rect(center=(WINDOW_WIDTH // 2, actions_y + i * 30))
            self.screen.blit(action_surface, action_rect)
        
        pygame.display.flip()
    
    def draw_simulation(self):
        """Draw the main simulation (extracted from draw method)"""
        self.screen.fill(BLACK)
        
        # Draw title
        title = self.font_large.render("The Twin Paradox - Complete Round-Trip Journey", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 25))
        self.screen.blit(title, title_rect)
        
        # Draw subtitle with current status
        if self.ship_b.has_returned:
            subtitle = "Journey complete! Compare the twins' ages."
            color = GREEN
        elif self.ship_b.journey_phase == JourneyPhase.TURNAROUND:
            subtitle = "Bob is turning around at maximum distance"
            color = ORANGE
        elif self.ship_b.journey_phase == JourneyPhase.INBOUND:
            subtitle = "Bob is returning to Earth"
            color = BLUE
        else:
            subtitle = "Bob is traveling away from Earth"
            color = RED
            
        subtitle_surface = self.font_medium.render(subtitle, True, color)
        subtitle_rect = subtitle_surface.get_rect(center=(WINDOW_WIDTH // 2, 50))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Draw spaceships
        self.draw_spaceship(self.ship_a)
        self.draw_spaceship(self.ship_b)
        
        # Draw journey timeline
        self.draw_journey_timeline()
        
        # Draw graph
        self.draw_graph()
        
        # Draw information panel
        self.draw_info_panel()
        
        # Draw distance indicator
        distance_y = 170
        distance_text = f"Distance from Earth: {self.ship_b.distance_traveled:.1f} light-years"
        distance_surface = self.font_small.render(distance_text, True, WHITE)
        self.screen.blit(distance_surface, (20, distance_y))
        
        # Draw max distance reference line
        max_dist_text = f"Maximum distance: {self.ship_b.max_distance:.1f} ly"
        max_dist_surface = self.font_small.render(max_dist_text, True, YELLOW)
        self.screen.blit(max_dist_surface, (20, distance_y + 20))
    
    def draw(self):
        """Draw the appropriate screen based on current state"""
        if self.show_config_screen:
            self.draw_config_screen()
        elif self.simulation_complete:
            self.draw_completion_screen()
        else:
            self.draw_simulation()
            pygame.display.flip()
    
    def run(self):
        """Main simulation loop"""
        running = True
        
        while running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            
            running = self.handle_events()
            self.update(dt)
            self.draw()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    simulation = TimeDialationSimulation()
    # Start with configuration screen
    simulation.show_config_screen = True
    simulation.run() 