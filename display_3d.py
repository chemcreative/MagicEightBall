import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time
import numpy as np
from OpenGL.GL import shaders
import pygame.freetype

class MagicEightBallDisplay3D:
    def __init__(self, width=800, height=800):
        # Initialize pygame and OpenGL
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Magic Eight Ball 3D - Use Arrow Keys to adjust position, +/- to adjust height")
        
        # Position adjustment variables
        self.position_y = -1.0  # Initial Y position
        self.position_x = 0.0   # Initial X position
        self.y_scale = 2.0      # Initial Y scale
        
        # Initialize font
        self.font = pygame.freetype.SysFont('Arial', 36)
        self.text = "What's on your mind?\nThe Magic Eight Ball knows."
        
        # Set up the 3D environment
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up light
        glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 10, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        # Set up the perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (width/height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 4, 5, 0, -0.5, 0, 0, 1, 0)
        
        # Animation properties
        self.start_time = time.time()
        self.opacity = 0.0
        self.rotation_angle = 0
        self.spin_complete = False
        self.spin_speed = 2.0
        self.target_angle = 0  # We'll set this to the nearest 90 degrees when stopping
        self.camera_distance = 5.0
        
        # Load the pyramid model
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        
        try:
            with open('pyramid.obj', 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('mtllib') or line.startswith('o') or line.startswith('s'):
                        continue
                        
                    # Remove inline comments
                    if '#' in line:
                        line = line[:line.index('#')].strip()
                        
                    if line.startswith('v '):  # Vertex
                        parts = line.split()[1:]
                        self.vertices.append([float(x) for x in parts])
                    elif line.startswith('vn '):  # Normal
                        parts = line.split()[1:]
                        self.normals.append([float(x) for x in parts])
                    elif line.startswith('vt '):  # Texture coordinate
                        parts = line.split()[1:]
                        self.texcoords.append([float(x) for x in parts])
                    elif line.startswith('f '):  # Face
                        parts = line.split()[1:]
                        face = []
                        for part in parts:
                            indices = part.split('/')
                            vertex_idx = int(indices[0]) - 1
                            face.append(vertex_idx)
                        self.faces.append(face)
            
            print(f"Loaded {len(self.vertices)} vertices")
            print(f"Loaded {len(self.normals)} normals")
            print(f"Loaded {len(self.faces)} faces")
            
        except Exception as e:
            print(f"Error loading pyramid model: {e}")
        
        self.clock = pygame.time.Clock()
        self.running = True

    def draw_text_overlay(self):
        if self.spin_complete:
            # Switch to 2D rendering
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            # Disable 3D features
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)
            
            # Render text
            text_surface, _ = self.font.render(self.text, (255, 255, 255))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            
            # Create and bind texture
            text_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, text_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_surface.get_width(), text_surface.get_height(),
                        0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
            # Draw text quad
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            x = (self.width - text_surface.get_width()) // 2
            y = (self.height - text_surface.get_height()) // 2
            
            glBegin(GL_QUADS)
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glTexCoord2f(0, 0); glVertex2f(x, y)
            glTexCoord2f(1, 0); glVertex2f(x + text_surface.get_width(), y)
            glTexCoord2f(1, 1); glVertex2f(x + text_surface.get_width(), y + text_surface.get_height())
            glTexCoord2f(0, 1); glVertex2f(x, y + text_surface.get_height())
            glEnd()
            
            # Clean up
            glDeleteTextures(1, [text_texture])
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
            
            # Restore 3D state
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            
            # Restore matrices
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    def draw_pyramid(self):
        if not self.vertices or not self.faces:
            return
            
        glPushMatrix()
        
        # Move the entire pyramid using the adjustable position
        glTranslatef(self.position_x, self.position_y, 0)
        
        # Scale the pyramid with taller tip
        glScalef(1.3, self.y_scale, 1.3)  # Use adjustable Y scale
        
        # Flip the model 180 degrees and adjust tilt
        glRotatef(180, 1, 0, 0)
        glRotatef(0, 0, 0, 1)  # Adjust tilt to be flat
        glRotatef(0, 0, 1, 0)  # Turn 45 degrees counter-clockwise
        
        # Apply rotation if still spinning
        if not self.spin_complete:
            glRotatef(self.rotation_angle, 0, 1, -0.1)
        else:
            # When stopped, use the target angle
            glRotatef(self.target_angle, 0, 1, -0.1)
            print(f"\n=== FINAL ROTATION STATE ===")
            print(f"Base rotations:")
            print(f"  - Flip: 180 degrees around X")
            print(f"  - Tilt: 0 degrees around Z")
            print(f"  - Turn: 0 degrees around Y")
            print(f"Final spin:")
            print(f"  - Angle: {self.target_angle} degrees")
            print(f"  - Axis: (0, 1, -0.1)")
            print(f"=======================\n")
        
        # Draw the main pyramid with current opacity
        glColor4f(0.2, 0.2, 0.8, self.opacity)  # Blue color with opacity
        
        glBegin(GL_QUADS)
        for face in self.faces:
            # Calculate face normal
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            
            vec1 = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]]
            vec2 = [v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]]
            
            normal = [
                vec1[1]*vec2[2] - vec1[2]*vec2[1],
                vec1[2]*vec2[0] - vec1[0]*vec2[2],
                vec1[0]*vec2[1] - vec1[1]*vec2[0]
            ]
            
            length = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
            if length > 0:
                normal = [n/length for n in normal]
                glNormal3f(*normal)
            
            for vertex_idx in face:
                glVertex3f(*self.vertices[vertex_idx])
        glEnd()
        
        # Draw edges
        glColor4f(0.1, 0.1, 0.4, self.opacity)
        glBegin(GL_LINES)
        for face in self.faces:
            for i in range(len(face)):
                v1 = self.vertices[face[i]]
                v2 = self.vertices[face[(i + 1) % len(face)]]
                glVertex3f(*v1)
                glVertex3f(*v2)
        glEnd()
        
        glPopMatrix()
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    # Adjust position with arrow keys
                    if event.key == pygame.K_UP:
                        self.position_y += 0.1
                    elif event.key == pygame.K_DOWN:
                        self.position_y -= 0.1
                    elif event.key == pygame.K_LEFT:
                        self.position_x -= 0.1
                    elif event.key == pygame.K_RIGHT:
                        self.position_x += 0.1
                    # Adjust height with +/- keys
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.y_scale += 0.1
                    elif event.key == pygame.K_MINUS:
                        self.y_scale -= 0.1
                    print(f"Position: X={self.position_x:.1f}, Y={self.position_y:.1f}, Height={self.y_scale:.1f}")
            
            current_time = time.time() - self.start_time
            
            if current_time >= 2.0:
                self.opacity = min(1.0, (current_time - 2.0) / 1.0)
            
            if not self.spin_complete:
                self.rotation_angle += self.spin_speed
                self.spin_speed *= 0.99
                
                # Print rotation info every 100 degrees
                if int(self.rotation_angle) % 100 == 0:
                    print(f"Current angle: {self.rotation_angle:.1f}, Speed: {self.spin_speed:.3f}")
                
                if self.rotation_angle >= 360 and self.spin_speed < 0.5:
                    # Calculate the nearest 90-degree angle and subtract 45 to compensate
                    self.target_angle = (round(self.rotation_angle / 90) * 90) - 45
                    # Smoothly transition to the target angle
                    self.rotation_angle = self.target_angle
                    self.spin_speed = 0
                    self.spin_complete = True
                    print(f"\n=== SPIN COMPLETE ===")
                    print(f"Final angle: {self.rotation_angle}")
                    print(f"Target angle: {self.target_angle}")
                    print(f"===================\n")
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            
            self.draw_pyramid()
            self.draw_text_overlay()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    display = MagicEightBallDisplay3D()
    display.run() 