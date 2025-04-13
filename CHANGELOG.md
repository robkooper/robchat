# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of RobChat
- Support for multiple document types (PDF, DOCX, TXT, HTML, PPTX, XLSX)
- RAG pipeline with document chunking and indexing
- FastAPI backend with RESTful API endpoints
- React-based frontend interface
- Configuration system with config.yaml
- Project-based document organization
- Source citation in responses
- Example folder with API usage demonstrations:
  - Python script (test_api.py) for API interaction
  - Shell script (test_api.sh) for API interaction
  - Sample documents about quantum computing and AI
  - Documentation on how to use the examples
- Static frontend implementation with vanilla JavaScript
- User authentication system with JWT tokens
- Theme selection with light, dark, and auto modes
- Enter key support for login form submission
- Comprehensive API documentation in README
- Detailed error response documentation
- New configuration options for authentication and RAG settings
- File upload and deletion test cases
- Users configuration file (users.yaml) with example template
- Application exit on missing users.yaml file

### Changed
- Updated password hashing to use bcrypt directly
- Improved error handling in authentication
- Fixed test cases to work with authentication
- Moved from React-based frontend to static HTML/JavaScript implementation
- Updated API endpoints to include user authentication
- Improved error handling and user feedback
- Enhanced login form with keyboard navigation support
- Reorganized test files into logical groups:
  - test_auth.py: Authentication tests
  - test_projects.py: Project access control tests
  - test_files.py: File operation tests
- Updated API documentation with detailed response formats and error codes
- Moved hardcoded values to configuration file:
  - Authentication settings (secret key, algorithm, token expiration)
  - RAG template for QA system
  - Vector store settings
- Improved password field mapping in user authentication
- Enhanced configuration documentation in README
- Added users.yaml to .gitignore for security

### Fixed
- Fixed password verification in authentication
- Fixed test cases to handle authentication properly
- Fixed file upload handling with authentication
- Fixed project switching with user authentication
- Fixed login form submission handling
- Fixed test assertions to match actual API responses
- Fixed error message consistency across endpoints
- Fixed user authentication by mapping password field to hashed_password
- Fixed configuration loading order in app initialization
- Fixed application startup to check for required configuration files

### Removed
- Removed React-based frontend components
- Removed frontend build system
- Removed unnecessary frontend dependencies
- Removed `/api/{user}/switch` endpoint and associated tests
- Removed redundant test cases 