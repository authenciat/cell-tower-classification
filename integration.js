import { TowerRenderer } from './towerRenderer.js';

// Create a container for the 3D view
const container = document.createElement('div');
container.style.width = '100%';
container.style.height = '400px'; // Adjust height as needed
document.body.appendChild(container);

// Initialize the tower renderer
const renderer = new TowerRenderer(
    container,
    container.clientWidth,
    container.clientHeight
);

// Example function to handle classification results
function onClassificationResult(towerType) {
    // Map classification labels to tower types
    const typeMap = {
        'lattice': 'lattice',
        'monopole': 'monopole',
        'guyed': 'guyed',
        'water_tower': 'water'
    };

    // Render the appropriate tower
    const mappedType = typeMap[towerType] || 'lattice'; // Default to lattice if unknown type
    renderer.renderTower(mappedType);
}

// Handle window resizing
window.addEventListener('resize', () => {
    renderer.resize(container.clientWidth, container.clientHeight);
});

// Example usage:
// When you get a classification result, call:
// onClassificationResult('lattice');
// This will render the appropriate tower type 