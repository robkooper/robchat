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
    
    const loginButton = document.getElementById('loginButton');
    const usernameField = document.getElementById('username');
    const passwordField = document.getElementById('password');
    
    // Function to update button state
    function updateButtonState() {
        const isValid = usernameField.value.trim() !== '' && passwordField.value.trim() !== '';
        loginButton.disabled = !isValid;
    }
    
    // Add input event listeners to update button state
    usernameField.addEventListener('input', updateButtonState);
    passwordField.addEventListener('input', updateButtonState);
    
    // Add Enter key handling for both fields
    function handleEnterKey(e) {
        if (e.key === 'Enter' && !loginButton.disabled) {
            handleLogin();
        }
    }
    
    usernameField.addEventListener('keypress', handleEnterKey);
    passwordField.addEventListener('keypress', handleEnterKey);
    
    // Handle login button click
    loginButton.addEventListener('click', function(e) {
        e.preventDefault();
        if (!loginButton.disabled) {
            handleLogin();
        }
    });
    
    // Handle form submission
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        if (!loginButton.disabled) {
            handleLogin();
        }
    });
    
    // Function to handle the actual login
    async function handleLogin() {
        const username = usernameField.value.trim();
        const password = passwordField.value.trim();
        
        if (!username || !password) {
            errorMessage.textContent = 'Please enter both username and password';
            errorMessage.style.display = 'block';
            return;
        }
        
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
    }
    
    // Initial button state
    updateButtonState();
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

    // Initialize login functionality
    initializeLogin();
}); 