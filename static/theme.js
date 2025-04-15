// Theme management
function setTheme(theme) {
    // Store the theme choice in localStorage
    localStorage.setItem('theme', theme);
    
    // Apply the actual theme (convert auto to system theme)
    const appliedTheme = theme === 'auto' ? getSystemTheme() : theme;
    document.documentElement.setAttribute('data-theme', appliedTheme);
    updateThemeIcon(appliedTheme);
    
    // Update active state of theme options
    document.querySelectorAll('.theme-selector[data-theme]').forEach(item => {
        if (item.getAttribute('data-theme') === theme) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

function getSystemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function updateThemeIcon(theme) {
    const icon = document.querySelector('#themeDropdown i');
    if (theme === 'auto') {
        theme = getSystemTheme();
    }
    icon.className = theme === 'dark' ? 'bi bi-moon' : 'bi bi-sun';
}

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme === 'auto') {
        const newTheme = getSystemTheme();
        document.documentElement.setAttribute('data-theme', newTheme);
        updateThemeIcon(newTheme);
    }
});

// Initialize theme
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'auto';
    setTheme(savedTheme);

    // Theme dropdown event listeners
    document.querySelectorAll('.theme-selector[data-theme]').forEach(item => {
        item.addEventListener('click', e => {
            e.preventDefault();
            e.stopPropagation();
            const theme = e.currentTarget.getAttribute('data-theme');
            setTheme(theme);
            // Close the dropdown after selection
            const dropdown = bootstrap.Dropdown.getInstance(document.getElementById('themeDropdown'));
            if (dropdown) {
                dropdown.hide();
            }
        });
    });
} 