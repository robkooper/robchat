// Helper function to get the auth token
function getAuthToken() {
    return localStorage.getItem('access_token');
}

// Helper function to check if user is authenticated
function isAuthenticated() {
    return !!getAuthToken();
}

// Helper function to make authenticated requests
async function fetchWithAuth(url, options = {}) {
    const token = getAuthToken();
    if (!token) {
        window.location.href = '/';
        return;
    }
    
    const headers = {
        'Authorization': `Bearer ${token}`,
        ...options.headers
    };
    
    const response = await fetch(url, {
        ...options,
        headers
    });
    
    if (response.status === 401) {
        // Token expired or invalid
        localStorage.removeItem('access_token');
        window.location.href = '/';
        return;
    }
    
    return response;
}

document.addEventListener('DOMContentLoaded', () => {
    // Check if user is authenticated
    if (!isAuthenticated()) {
        window.location.href = '/';
        return;
    }

    // Initialize theme
    initializeTheme();

    // Get user details from JWT token
    const token = getAuthToken();
    const tokenPayload = JSON.parse(atob(token.split('.')[1]));
    document.getElementById('usernameDisplay').textContent = tokenPayload.fullname;
    
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatMessages = document.getElementById('chatMessages');
    const addFileBtn = document.getElementById('addFileBtn');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const logoutBtn = document.getElementById('logoutBtn');
    const userDropdown = document.getElementById('userDropdown');
    const projectDropdown = document.getElementById('projectDropdown');
    const projectList = document.getElementById('projectList');
    const addProjectBtn = document.getElementById('addProjectBtn');
    const createProjectBtn = document.getElementById('createProjectBtn');
    const projectNameInput = document.getElementById('projectName');
    const newProjectModal = new bootstrap.Modal(document.getElementById('newProjectModal'));
    const newProjectModalElement = document.getElementById('newProjectModal');

    // Get username from the JWT token
    const username = tokenPayload.sub;

    // Initialize currentProject before using it
    let currentProject = "";

    // Clear input and set focus when modal is shown
    newProjectModalElement.addEventListener('shown.bs.modal', () => {
        projectNameInput.value = '';
        projectNameInput.focus();
    });

    // Set focus back to message input when modal is hidden
    newProjectModalElement.addEventListener('hide.bs.modal', () => {
        projectNameInput.value = '';
        // Remove focus from the input before the modal starts hiding
        projectNameInput.blur();
    });

    newProjectModalElement.addEventListener('hidden.bs.modal', () => {
        // Use requestAnimationFrame to ensure the modal is fully hidden
        requestAnimationFrame(() => {
            messageInput.focus();
        });
    });

    // Enable all tooltips
    const tooltipList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipList.map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl, {
        trigger: 'hover'
    }));

    // Load projects and files on startup
    loadProjects().then(() => {
        // Initialize dropdown after projects are loaded
        const dropdown = new bootstrap.Dropdown(projectDropdown);
    });

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    addFileBtn.addEventListener('click', (e) => {
        if (!currentProject) {
            alert('Please select a project before uploading files.');
            return;
        }
        
        // Prevent event propagation
        e.preventDefault();
        e.stopPropagation();
        
        // Use the existing file input
        fileInput.click();
    });

    // Add change event listener to the existing file input
    fileInput.addEventListener('change', (event) => {
        handleFileUpload(event);
    });

    logoutBtn.addEventListener('click', handleLogout);
    createProjectBtn.addEventListener('click', createNewProject);
    projectNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            createNewProject();
        }
    });

    addProjectBtn.addEventListener('click', (e) => {
        e.preventDefault();
        newProjectModal.show();
    });

    function setInputLoading(isLoading) {
        messageInput.disabled = isLoading;
        sendBtn.disabled = isLoading;
        
        if (isLoading) {
            messageInput.value = 'AI is thinking';
            messageInput.classList.add('loading-dots');
        } else {
            messageInput.classList.remove('loading-dots');
        }
    }

    function addMessage(text, sender, isThinking = false) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';

        // Add avatar for bot messages
        if (sender === 'bot') {
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = 'AI';
            messageContainer.appendChild(avatar);
        }

        const message = document.createElement('div');
        message.className = `message ${sender === 'user' ? 'user-message' : 'bot-message'} ${isThinking ? 'thinking' : ''}`;
        
        if (isThinking) {
            const dots = document.createElement('span');
            dots.className = 'thinking-dots';
            message.appendChild(dots);
        } else {
            message.textContent = text;
        }

        messageContainer.appendChild(message);

        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-time';
        const now = new Date();
        timestamp.textContent = now.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
        messageContainer.appendChild(timestamp);

        chatMessages.appendChild(messageContainer);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageContainer;
    }

    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message || !currentProject) return;

        // Clear input and add user message
        messageInput.value = '';
        addMessage(message, 'user');

        // Add thinking message
        const thinkingMessage = addMessage('', 'bot', true);

        try {
            const response = await fetchWithAuth(`/api/${encodeURIComponent(username)}/${encodeURIComponent(currentProject)}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: message
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Remove thinking message and add actual response with current timestamp
            thinkingMessage.remove();
            const now = new Date();
            const timestamp = now.toLocaleTimeString('en-US', { 
                hour: 'numeric', 
                minute: '2-digit',
                hour12: true 
            });
            addMessage(data.answer, 'bot');
        } catch (error) {
            console.error('Error:', error);
            thinkingMessage.remove();
            const now = new Date();
            const timestamp = now.toLocaleTimeString('en-US', { 
                hour: 'numeric', 
                minute: '2-digit',
                hour12: true 
            });
            addMessage('Sorry, there was an error processing your request.', 'bot');
        }
    }

    async function loadProjects() {
        try {
            const response = await fetchWithAuth(`/api/${encodeURIComponent(username)}/projects`);
            const data = await response.json();
            
            // Clear existing projects
            projectList.innerHTML = '';
            
            // Update current project
            currentProject = data.current_project || "default";
            projectDropdown.textContent = currentProject;
            
            // Add all projects to the dropdown, including current project
            const projects = data.projects || [];
            if (currentProject && !projects.includes(currentProject)) {
                projects.unshift(currentProject);
            }
            
            // Add projects to dropdown
            projects.forEach(project => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.className = 'dropdown-item';
                a.href = '#';
                a.textContent = project;
                a.onclick = () => switchProject(project);
                li.appendChild(a);
                projectList.appendChild(li);
            });

            // Load files after currentProject is set
            await loadFiles();
            
        } catch (error) {
            console.error('Error loading projects:', error);
        }
    }

    async function switchProject(project) {
        try {
            currentProject = project;
            projectDropdown.textContent = project;
            await loadFiles();
        } catch (error) {
            console.error('Error switching project:', error);
        }
    }

    async function loadFiles() {
        if (!currentProject) return;
        
        try {
            const response = await fetchWithAuth(`/api/${encodeURIComponent(username)}/${encodeURIComponent(currentProject)}/files`);
            if (!response.ok) {
                console.error('Error loading files:', response.status, response.statusText);
                displayFiles([]); // Display empty list on error
                return;
            }
            const data = await response.json();
            displayFiles(data.files || []);
        } catch (error) {
            console.error('Error loading files:', error);
            displayFiles([]); // Display empty list on error
        }
    }

    function displayFiles(files) {
        if (!files || !Array.isArray(files)) {
            console.error('Invalid files data:', files);
            files = [];
        }
        
        fileList.innerHTML = '';
        
        files.forEach(file => {
            const tr = document.createElement('tr');
            
            // Filename cell
            const filenameTd = document.createElement('td');
            filenameTd.textContent = file.filename || file;
            tr.appendChild(filenameTd);
            
            // Actions cell
            const actionsTd = document.createElement('td');
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.innerHTML = '<i class="bi bi-trash"></i>';
            deleteBtn.onclick = () => deleteFile(file.filename || file);
            actionsTd.appendChild(deleteBtn);
            tr.appendChild(actionsTd);
            
            fileList.appendChild(tr);
        });
    }

    async function deleteFile(filename) {
        if (!currentProject) return;
        
        try {
            const response = await fetchWithAuth(`/api/${encodeURIComponent(username)}/${encodeURIComponent(currentProject)}/files/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Reload files after successful deletion
            await loadFiles();
        } catch (error) {
            console.error('Error deleting file:', error);
            alert('Failed to delete file. Please try again.');
        }
    }

    async function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file || !currentProject) {
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetchWithAuth(`/api/${encodeURIComponent(username)}/${encodeURIComponent(currentProject)}/files`, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                loadFiles();
            } else {
                console.error('Error uploading file:', await response.text());
            }
        } catch (error) {
            console.error('Error in file upload:', error);
        }

        // Reset file input
        fileInput.value = '';
    }

    function handleLogout() {
        // Remove the token and reload the page
        localStorage.removeItem('access_token');
        window.location.href = '/';
    }

    async function createNewProject() {
        const projectName = projectNameInput.value.trim();
        if (!projectName) return;

        try {
            // Clear the input and close the modal
            projectNameInput.value = '';
            const modal = bootstrap.Modal.getInstance(document.getElementById('newProjectModal'));
            modal.hide();
            
            // Set as current project
            currentProject = projectName;
            projectDropdown.textContent = projectName;
            
            // Add to project list
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.className = 'dropdown-item active';
            a.href = '#';
            a.textContent = projectName;
            a.addEventListener('click', (e) => {
                e.preventDefault();
                switchProject(projectName);
                projectDropdown.textContent = projectName;
                // Remove active class from all items
                document.querySelectorAll('#projectList .dropdown-item').forEach(item => {
                    item.classList.remove('active');
                });
                // Add active class to clicked item
                a.classList.add('active');
            });
            li.appendChild(a);
            projectList.insertBefore(li, projectList.firstChild);
            
            // Remove active class from all other items
            document.querySelectorAll('#projectList .dropdown-item').forEach(item => {
                if (item !== a) {
                    item.classList.remove('active');
                }
            });
            
            // Clear and reload files for the new project
            fileList.innerHTML = '';
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to create project. Please try again.');
        }
    }
}); 