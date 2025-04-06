import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export class TowerRenderer {
    constructor(container, width, height) {
        this.container = container;
        this.width = width;
        this.height = height;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        
        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setClearColor(0xf0f0f0); // Light background
        container.appendChild(this.renderer.domElement);
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(ambientLight);
        this.scene.add(directionalLight);
        
        // Add controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        
        // Add ground
        this.addGround();
        
        // Start animation
        this.animate();
    }

    // Helper function to create struts
    createStrut(start, end, radius) {
        const direction = new THREE.Vector3().subVectors(end, start);
        const height = direction.length();
        const geometry = new THREE.CylinderGeometry(radius, radius, height, 8);
        const material = new THREE.MeshPhongMaterial({ 
            color: 0xD0D0D0,
            metalness: 0.9,
            roughness: 0.2
        });
        const strut = new THREE.Mesh(geometry, material);
        
        // Position and rotate the strut
        strut.position.copy(start);
        strut.position.addScaledVector(direction, 0.5);
        strut.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            direction.normalize()
        );
        
        return strut;
    }

    addGround() {
        const groundGeometry = new THREE.PlaneGeometry(200, 200);
        const groundMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xE0F0E0,
            metalness: 0.1,
            roughness: 0.8,
            side: THREE.DoubleSide
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.1;
        this.scene.add(ground);
    }

    // Tower creation methods
    createLatticeTower(height = 40) {
        const tower = new THREE.Group();
        const baseWidth = 6;
        const topWidth = 1.2;
        const segments = 12;
        
        // Create vertical corner poles
        for (let i = 0; i < 4; i++) {
            const x = ((i % 2) * 2 - 1) * baseWidth / 2;
            const z = (Math.floor(i / 2) * 2 - 1) * baseWidth / 2;
            const topX = ((i % 2) * 2 - 1) * topWidth / 2;
            const topZ = (Math.floor(i / 2) * 2 - 1) * topWidth / 2;
            
            const pole = this.createStrut(
                new THREE.Vector3(x, 0, z),
                new THREE.Vector3(topX, height, topZ),
                0.2
            );
            tower.add(pole);
        }
        
        // Add cross bracing
        for (let i = 0; i < segments; i++) {
            const y = (i / segments) * height;
            const nextY = ((i + 1) / segments) * height;
            const width = baseWidth - (baseWidth - topWidth) * (y / height);
            const nextWidth = baseWidth - (baseWidth - topWidth) * (nextY / height);
            
            // Horizontal supports
            for (let j = 0; j < 4; j++) {
                const x1 = ((j % 2) * 2 - 1) * width / 2;
                const z1 = (Math.floor(j / 2) * 2 - 1) * width / 2;
                const x2 = ((((j + 1) % 4) % 2) * 2 - 1) * width / 2;
                const z2 = (Math.floor(((j + 1) % 4) / 2) * 2 - 1) * width / 2;
                
                const support = this.createStrut(
                    new THREE.Vector3(x1, y, z1),
                    new THREE.Vector3(x2, y, z2),
                    0.1
                );
                tower.add(support);
            }
            
            // Diagonal bracing
            if (i < segments - 1) {
                for (let j = 0; j < 4; j++) {
                    const x1 = ((j % 2) * 2 - 1) * width / 2;
                    const z1 = (Math.floor(j / 2) * 2 - 1) * width / 2;
                    const x2 = ((((j + 1) % 4) % 2) * 2 - 1) * nextWidth / 2;
                    const z2 = (Math.floor(((j + 1) % 4) / 2) * 2 - 1) * nextWidth / 2;
                    
                    const brace = this.createStrut(
                        new THREE.Vector3(x1, y, z1),
                        new THREE.Vector3(x2, nextY, z2),
                        0.1
                    );
                    tower.add(brace);
                }
            }
        }
        
        return tower;
    }

    createMonolithTower(height = 40) {
        const tower = new THREE.Group();
        
        // Create main pole
        const poleGeometry = new THREE.CylinderGeometry(0.8, 0.8, height, 16);
        const poleMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xD0D0D0,
            metalness: 0.9,
            roughness: 0.2
        });
        const pole = new THREE.Mesh(poleGeometry, poleMaterial);
        pole.position.y = height / 2;
        tower.add(pole);
        
        // Add antenna arrays in top 15%
        const levels = 3;
        const topSection = height * 0.15;
        const levelSpacing = topSection / 3;
        const startHeight = height - (levelSpacing * 0.5);
        
        const antennaMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xFFFFFF,
            metalness: 0.8,
            roughness: 0.2
        });
        
        for (let level = 0; level < levels; level++) {
            const levelHeight = startHeight - (level * levelSpacing);
            const numPanels = 6;
            const radius = 1.2;
            
            // Create support ring
            const ringGeometry = new THREE.TorusGeometry(radius, 0.05, 8, 24);
            const ring = new THREE.Mesh(ringGeometry, poleMaterial);
            ring.position.y = levelHeight;
            ring.rotation.x = Math.PI / 2;
            tower.add(ring);
            
            for (let i = 0; i < numPanels; i++) {
                const angle = (i / numPanels) * Math.PI * 2;
                const x = Math.cos(angle) * radius;
                const z = Math.sin(angle) * radius;
                
                // Panel dimensions
                const panelWidth = 0.4;
                const panelHeight = 2.0;
                const panelDepth = 0.2;
                
                const panel = new THREE.Mesh(
                    new THREE.BoxGeometry(panelWidth, panelHeight, panelDepth),
                    antennaMaterial
                );
                
                panel.position.set(x, levelHeight, z);
                panel.lookAt(new THREE.Vector3(0, levelHeight, 0));
                tower.add(panel);
                
                // Add mounting bracket
                const bracket = new THREE.Mesh(
                    new THREE.BoxGeometry(0.2, 0.4, 0.6),
                    poleMaterial
                );
                bracket.position.set(x * 0.8, levelHeight, z * 0.8);
                bracket.lookAt(new THREE.Vector3(0, levelHeight, 0));
                tower.add(bracket);
                
                // Add cables
                const cable = this.createStrut(
                    new THREE.Vector3(x * 0.6, levelHeight, z * 0.6),
                    new THREE.Vector3(x, levelHeight, z),
                    0.03
                );
                tower.add(cable);
            }
        }
        
        return tower;
    }

    createGuyedTower(height = 40) {
        const tower = new THREE.Group();
        
        // Create main pole
        const poleGeometry = new THREE.CylinderGeometry(0.4, 0.4, height, 16);
        const poleMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xD0D0D0,
            metalness: 0.9,
            roughness: 0.2
        });
        const pole = new THREE.Mesh(poleGeometry, poleMaterial);
        pole.position.y = height / 2;
        tower.add(pole);
        
        // Add guy-wires
        const wireHeights = [height * 0.75, height * 0.5, height * 0.25];
        const numWires = 4;
        const anchorDistance = height * 0.8;
        
        const wireMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x888888,
            metalness: 0.95,
            roughness: 0.1
        });
        
        wireHeights.forEach(wireHeight => {
            for (let i = 0; i < numWires; i++) {
                const angle = (i / numWires) * Math.PI * 2;
                const anchorX = Math.cos(angle) * anchorDistance;
                const anchorZ = Math.sin(angle) * anchorDistance;
                
                // Create guy-wire
                const wire = this.createStrut(
                    new THREE.Vector3(0, wireHeight, 0),
                    new THREE.Vector3(anchorX, 0, anchorZ),
                    0.03
                );
                wire.material = wireMaterial;
                tower.add(wire);
                
                // Create anchor point
                const anchor = new THREE.Mesh(
                    new THREE.BoxGeometry(0.5, 0.5, 0.5),
                    poleMaterial
                );
                anchor.position.set(anchorX, 0.25, anchorZ);
                tower.add(anchor);
                
                // Create attachment point sphere
                const attachment = new THREE.Mesh(
                    new THREE.SphereGeometry(0.1),
                    wireMaterial
                );
                attachment.position.set(0, wireHeight, 0);
                tower.add(attachment);
            }
        });
        
        // Add antennas at top
        const numAntennas = 6;
        const antennaRadius = 1.2;
        
        const antennaMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xFFFFFF,
            metalness: 0.8,
            roughness: 0.2
        });
        
        for (let i = 0; i < numAntennas; i++) {
            const angle = (i / numAntennas) * Math.PI * 2;
            const x = Math.cos(angle) * antennaRadius;
            const z = Math.sin(angle) * antennaRadius;
            
            const antenna = new THREE.Mesh(
                new THREE.BoxGeometry(0.3, 2.0, 0.2),
                antennaMaterial
            );
            antenna.position.set(x, height - 1, z);
            antenna.lookAt(new THREE.Vector3(0, height - 1, 0));
            tower.add(antenna);
            
            // Add connecting lines
            const connector = this.createStrut(
                new THREE.Vector3(0, height - 1, 0),
                new THREE.Vector3(x, height - 1, z),
                0.02
            );
            connector.material = wireMaterial;
            tower.add(connector);
            
            const verticalSupport = this.createStrut(
                new THREE.Vector3(x * 0.5, height - 2, z * 0.5),
                new THREE.Vector3(x, height - 1, z),
                0.02
            );
            verticalSupport.material = wireMaterial;
            tower.add(verticalSupport);
        }
        
        return tower;
    }

    createWaterTower(height = 40) {
        const tower = new THREE.Group();
        
        // Water tower parameters
        const tankRadius = 8;
        const tankHeight = 6;
        const numColumns = 6;
        const columnHeight = height * 0.6;
        
        // Create the main water tank
        const tankGeometry = new THREE.SphereGeometry(tankRadius, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.6);
        const tankMaterial = new THREE.MeshPhongMaterial({
            color: 0xE0E0E0,
            metalness: 0.85,
            roughness: 0.15
        });
        const tank = new THREE.Mesh(tankGeometry, tankMaterial);
        tank.position.y = columnHeight;
        tower.add(tank);
        
        // Add support columns
        for (let i = 0; i < numColumns; i++) {
            const angle = (i / numColumns) * Math.PI * 2;
            const x = Math.cos(angle) * tankRadius * 0.7;
            const z = Math.sin(angle) * tankRadius * 0.7;
            
            const column = this.createStrut(
                new THREE.Vector3(x, 0, z),
                new THREE.Vector3(x, columnHeight, z),
                0.3
            );
            tower.add(column);
            
            // Add diagonal supports
            if (i < numColumns) {
                const nextAngle = ((i + 1) / numColumns) * Math.PI * 2;
                const nextX = Math.cos(nextAngle) * tankRadius * 0.7;
                const nextZ = Math.sin(nextAngle) * tankRadius * 0.7;
                
                const support = this.createStrut(
                    new THREE.Vector3(x, columnHeight * 0.6, z),
                    new THREE.Vector3(nextX, columnHeight * 0.8, nextZ),
                    0.2
                );
                tower.add(support);
            }
        }
        
        return tower;
    }

    // Main rendering methods
    renderTower(type) {
        // Clear existing tower
        this.clearScene();
        this.addGround();
        
        let tower;
        switch(type.toLowerCase()) {
            case 'lattice':
                tower = this.createLatticeTower();
                this.camera.position.set(30, 40, 30);
                break;
            case 'monopole':
            case 'monolith':
                tower = this.createMonolithTower();
                this.camera.position.set(30, 40, 30);
                break;
            case 'guyed':
                tower = this.createGuyedTower();
                this.camera.position.set(60, 40, 60);
                break;
            case 'water':
                tower = this.createWaterTower();
                this.camera.position.set(60, 40, 60);
                break;
            default:
                console.error('Unknown tower type:', type);
                return;
        }
        
        this.scene.add(tower);
        this.camera.lookAt(0, 20, 0);
        this.controls.update();
    }

    clearScene() {
        while(this.scene.children.length > 0) { 
            this.scene.remove(this.scene.children[0]); 
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    // Resize handler
    resize(width, height) {
        this.width = width;
        this.height = height;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
} 