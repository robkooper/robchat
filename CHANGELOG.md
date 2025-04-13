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

### Changed
- Updated password hashing to use bcrypt directly
- Improved error handling in authentication
- Fixed test cases to work with authentication
- Moved from React-based frontend to static HTML/JavaScript implementation
- Updated API endpoints to include user authentication
- Improved error handling and user feedback
- Enhanced login form with keyboard navigation support

### Fixed
- Fixed password verification in authentication
- Fixed test cases to handle authentication properly
- Fixed file upload handling with authentication
- Fixed project switching with user authentication
- Fixed login form submission handling

### Removed
- Removed React-based frontend components
- Removed frontend build system
- Removed unnecessary frontend dependencies

## [1.0.0]

### Added
- Initial release of RobChat
- Support for multiple document types (PDF, DOCX, TXT, HTML, PPTX, XLSX)
- RAG pipeline with document chunking and indexing
- FastAPI backend with RESTful API endpoints
- React-based frontend interface
- Configuration system with config.yaml
- Project-based document organization
- Source citation in responses 