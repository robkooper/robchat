// Theme management
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
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
    if (localStorage.getItem('theme') === 'auto') {
        setTheme('auto');
    }
});

// Initialize login functionality
function initializeLogin() {
    const loginForm = document.getElementById('loginForm');
    if (!loginForm) return;
    
    const errorMessage = document.getElementById('errorMessage');
    if (!errorMessage) return;
    
    // Add Enter key handling for both fields
    const usernameField = document.getElementById('username');
    const passwordField = document.getElementById('password');
    
    function handleEnterKey(e) {
        if (e.key === 'Enter' && usernameField.value && passwordField.value) {
            loginForm.dispatchEvent(new Event('submit'));
        }
    }
    
    usernameField.addEventListener('keypress', handleEnterKey);
    passwordField.addEventListener('keypress', handleEnterKey);
    
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        
        // Hide any previous error message
        errorMessage.style.display = 'none';
        
        try {
            const formData = new URLSearchParams({
                'username': username,
                'password': password
            });
            
            const response = await fetch('/token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                localStorage.setItem('access_token', data.access_token);
                window.location.href = '/chat';
            } else {
                const errorData = await response.json();
                errorMessage.textContent = errorData.detail || 'Login failed. Please check your credentials.';
                errorMessage.style.display = 'block';
            }
        } catch (error) {
            errorMessage.textContent = 'An error occurred during login. Please try again.';
            errorMessage.style.display = 'block';
        }
    });
}

// Initialize everything when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme
    const savedTheme = localStorage.getItem('theme') || 'auto';
    setTheme(savedTheme);

    // Theme dropdown event listeners
    document.querySelectorAll('[data-theme]').forEach(item => {
        item.addEventListener('click', e => {
            e.preventDefault();
            const theme = e.currentTarget.getAttribute('data-theme');
            setTheme(theme);
        });
    });
    
    // Initialize login
    initializeLogin();
}); 