# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- TL;DR section in README for quick setup
- Clarification about configuration files in README
- Support for multiple document types (PDF, DOCX, TXT, HTML, PPTX, XLSX)
- RAG pipeline with document chunking and indexing
- FastAPI backend with RESTful API endpoints
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
- Updated ChromaDB to use a single collection named "robchat" instead of per-user/project collections
- Added Bootstrap and Vanilla JavaScript references to README
- Added detailed logging to login functionality for debugging
- Added comprehensive workflow test suite for file upload, querying, and deletion
- GitHub Actions workflows for automated testing and linting:
  - Parallel pytest workflow with Python 3.9, 3.10, and 3.11 support
  - Pylint workflow with minimum score requirement
  - Test coverage reporting and Codecov integration

### Changed
- Updated README to clarify configuration file setup process
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
- Improved login form submission handling
- Cleaned up debug code and console.log statements
- Removed hardcoded localhost URLs
- Improved error handling to be more graceful
- Fixed username extraction from JWT token

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
- Fixed login form submission not triggering
- Fixed username undefined error in project loading
- Fixed Enter key handling in login form

### Removed
- Removed React-based frontend components
- Removed frontend build system
- Removed unnecessary frontend dependencies
- Removed `/api/{user}/switch` endpoint and associated tests
- Removed redundant test cases
- Removed PyPDF2 dependency in favor of pypdf 