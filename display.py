import pygame
import math
import os
import random
import time
import sys

# Initialize pygame
pygame.init()

class MagicEightBallDisplay:
    def __init__(self, width=800, height=800):
        # Basic screen setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Magic Eight Ball")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.TRIANGLE_BLUE = (0, 0, 180)  # Base blue for the triangle
        self.SHADOW_BLUE = (0, 0, 100)    # Darker blue for shadow
        self.HIGHLIGHT_BLUE = (0, 0, 220) # Lighter blue for highlight
        
        # Font - make it bigger
        self.font = pygame.font.Font(None, 36)  # Reduced from 48 to 36
        
        # Text properties
        self.current_text = ""
        self.text_alpha = 0  # For fade effect
        
        # Animation properties
        self.reveal_alpha = 0
        self.start_time = time.time()
        self.animation_phase = "fade_in"  # fade_in -> wait -> spin -> reveal
        self.wave_offset = 0  # For wave animation
        
        # 3D animation properties
        self.rotation_angle = 0
        self.scale_factor = 1.0
        self.perspective_skew = 0
        self.spin_direction = 1
        self.spin_speed = 0
        self.flip_progress = 0
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Running state
        self.running = True
        self.animation_complete = False
    
    def calculate_triangle_bounds(self, center, size, rotation=0, scale=1.0, skew=0):
        """Calculate the bounding box of the triangle for text fitting with 3D effects"""
        height = size * 0.9  # Triangle height
        width = size * 0.9   # Triangle width
        
        # Calculate base points
        base_points = [
            (center[0], center[1] + height/2),  # Bottom point
            (center[0] - width/2, center[1] - height/2),  # Top left
            (center[0] + width/2, center[1] - height/2)   # Top right
        ]
        
        # Apply 3D transformations
        points = []
        for x, y in base_points:
            # Apply rotation
            dx = x - center[0]
            dy = y - center[1]
            rotated_x = dx * math.cos(rotation) - dy * math.sin(rotation)
            rotated_y = dx * math.sin(rotation) + dy * math.cos(rotation)
            
            # Apply scale
            scaled_x = rotated_x * scale
            scaled_y = rotated_y * scale
            
            # Apply perspective skew
            skewed_x = scaled_x + (scaled_y * skew)
            
            # Return to center
            final_x = skewed_x + center[0]
            final_y = scaled_y + center[1]
            
            points.append((final_x, final_y))
        
        # Calculate the bounding box
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        
        return {
            'points': points,
            'bounds': (min_x, min_y, max_x - min_x, max_y - min_y),
            'center': center,
            'top': min_y,
            'bottom': max_y
        }
    
    def get_wave_alpha(self, x, y, base_alpha):
        # Create a more dramatic wave pattern
        wave = math.sin((x * 0.05) + self.wave_offset) * 0.5 + 0.5  # Increased frequency
        wave = wave * 0.9 + 0.1  # More contrast (10% to 100% opacity)
        
        # Add a second wave with different frequency
        wave2 = math.sin((x * 0.03) + (y * 0.02) + self.wave_offset * 1.5) * 0.4 + 0.6
        wave = (wave + wave2) * 0.5
        
        # Add a third wave for more complexity
        wave3 = math.sin((x * 0.02) + (y * 0.03) + self.wave_offset * 0.8) * 0.3 + 0.7
        wave = (wave + wave3) * 0.5
        
        # Apply the wave to the base alpha with more dramatic effect
        return int(base_alpha * wave)
    
    def draw_ball(self):
        # Draw the main circle (Magic Eight Ball)
        center = (self.width // 2, self.height // 2)
        radius = min(self.width, self.height) // 2 - 20
        
        # Draw the window (slightly smaller circle)
        window_radius = radius * 0.7  # Made window smaller
        window_center = (center[0], center[1] - radius * 0.1)  # Slightly higher center
        
        # Draw window background
        pygame.draw.circle(self.screen, self.BLACK, window_center, window_radius)
        
        # Calculate triangle bounds with 3D effects
        triangle_size = window_radius * 2.2
        triangle_info = self.calculate_triangle_bounds(
            window_center, 
            triangle_size,
            self.rotation_angle,
            self.scale_factor,
            self.perspective_skew
        )
        
        # Create a surface for the triangle
        triangle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw shadow triangle
        shadow_offset = 5
        shadow_points = [(x + shadow_offset, y + shadow_offset) for x, y in triangle_info['points']]
        pygame.draw.polygon(triangle_surface, (*self.SHADOW_BLUE, self.reveal_alpha), shadow_points)
        self.screen.blit(triangle_surface, (0, 0))
        
        # Create a new surface for the main triangle
        triangle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw main triangle
        pygame.draw.polygon(triangle_surface, (*self.TRIANGLE_BLUE, self.reveal_alpha), triangle_info['points'])
        self.screen.blit(triangle_surface, (0, 0))
        
        # Draw highlight on top edge
        highlight_points = [
            triangle_info['points'][1],  # Top left
            triangle_info['points'][2],  # Top right
            (triangle_info['points'][2][0] - 20, triangle_info['points'][2][1] + 20),  # Bottom right offset
            (triangle_info['points'][1][0] + 20, triangle_info['points'][1][1] + 20)   # Bottom left offset
        ]
        
        highlight_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.polygon(highlight_surface, (*self.HIGHLIGHT_BLUE, self.reveal_alpha), highlight_points)
        self.screen.blit(highlight_surface, (0, 0))
        
        return triangle_info
    
    def draw_text(self, triangle_info):
        if self.current_text:
            # Get the bounding box of the triangle
            min_x, min_y, width, height = triangle_info['bounds']
            center = triangle_info['center']
            
            # Calculate text area (80% of triangle size)
            text_width = width * 0.8
            text_height = height * 0.8
            
            # Calculate maximum width for text (90% of text area width)
            max_width = text_width * 0.9
            
            # Split text into multiple lines if needed
            words = self.current_text.split()
            lines = []
            current_line = []
            
            # Create lines that fit within the text area
            for word in words:
                test_line = ' '.join(current_line + [word])
                test_surface = self.font.render(test_line, True, self.WHITE)
                if test_surface.get_width() < max_width:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            # Calculate total text height
            total_height = len(lines) * self.font.get_height()
            
            # Calculate text position (moved up 55 pixels)
            text_y = center[1] - (len(lines) * total_height) / 2 - 55  # Changed from -25 to -55
            
            # Draw each line with wave effect
            for i, line in enumerate(lines):
                text_surface = self.font.render(line, True, self.WHITE)
                if self.animation_phase == "fade_in":
                    # Apply wave-based alpha to text
                    alpha = self.get_wave_alpha(center[0], text_y + i * self.font.get_height(), self.text_alpha)
                    text_surface.set_alpha(alpha)
                else:
                    text_surface.set_alpha(self.text_alpha)
                text_rect = text_surface.get_rect(center=(center[0], text_y + i * self.font.get_height()))
                self.screen.blit(text_surface, text_rect)
    
    def update_animation(self):
        current_time = time.time() - self.start_time
        
        if self.animation_phase == "fade_in":
            # Simple 2-second fade
            self.reveal_alpha = min(255, self.reveal_alpha + 2.125)  # 255/120 frames ≈ 2.125 per frame for 2 seconds
            self.text_alpha = self.reveal_alpha  # Sync text alpha with triangle
            if self.reveal_alpha >= 255:
                self.animation_phase = "wait"
        
        elif self.animation_phase == "wait":
            # Do nothing, wait for external trigger
            pass
        
        elif self.animation_phase == "spin":
            # Update 3D effects
            self.rotation_angle += self.spin_speed * self.spin_direction
            self.spin_speed += 0.005  # Gradually increase spin speed
            
            # Add perspective skew
            self.perspective_skew = math.sin(self.rotation_angle) * 0.3
            
            # Scale effect
            self.scale_factor = 1.0 + math.sin(self.rotation_angle * 2) * 0.2
            
            # Change direction and slow down
            if self.spin_speed > 0.3:
                self.spin_direction *= -1
                self.spin_speed *= 0.95
            
            # Transition to reveal when animation has slowed down
            if self.spin_speed < 0.05:
                self.animation_phase = "reveal"
                self.rotation_angle = 0
                self.scale_factor = 1.0
                self.perspective_skew = 0
                self.spin_speed = 0
        
        elif self.animation_phase == "reveal":
            self.reveal_alpha = min(255, self.reveal_alpha + 2.125)  # Same fade speed as fade_in
            self.text_alpha = self.reveal_alpha  # Sync text alpha with triangle
            if self.reveal_alpha >= 255:
                self.animation_phase = "done"
    
    def update_text(self, new_text):
        self.current_text = new_text
        self.text_alpha = 0  # Reset alpha for fade effect
    
    def run(self):
        while self.running and not self.animation_complete:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            # Update animation
            self.update_animation()
            
            # Update text alpha for fade effect
            if self.animation_phase == "done" and self.current_text and self.text_alpha < 255:
                self.text_alpha = min(255, self.text_alpha + 5)
                if self.text_alpha >= 255:
                    # Wait a bit before closing
                    time.sleep(2)  # Show the text for 2 seconds
                    self.animation_complete = True
            
            # Clear screen
            self.screen.fill(self.BLACK)
            
            # Draw ball and get triangle info
            triangle_info = self.draw_ball()
            
            # Draw text using triangle bounds
            self.draw_text(triangle_info)
            
            # Update display
            pygame.display.flip()
            
            # Cap at 30 FPS
            self.clock.tick(30)
        
        pygame.quit()
        # Ensure the process exits
        os._exit(0)

def show_response(text):
    display = MagicEightBallDisplay()
    display.update_text(text)
    display.run()

if __name__ == "__main__":
    # If text is provided as command line argument, use it
    if len(sys.argv) > 1:
        # Remove any extra quotes that might have been added
        text = sys.argv[1].strip('"')
    else:
        text = "What's on your mind? The Magic Eight Ball knows."
    
    show_response(text) 