/**
 * PDF Question Extractor Frontend Application
 * 
 * Main JavaScript application for the PDF Question Extractor web interface.
 * Handles file uploads, real-time processing updates, data grid management,
 * and user interactions.
 * 
 * Features:
 * - File upload with drag & drop
 * - WebSocket real-time updates
 * - Tabulator.js data grid
 * - Auto-save functionality
 * - Bulk operations
 * - Export functionality
 * - Toast notifications
 * - Responsive design
 */

class PDFQuestionExtractorApp {
    constructor() {
        // Application state
        this.currentPage = 1;
        this.perPage = 20;
        this.totalQuestions = 0;
        this.selectedRows = [];
        this.autoSaveTimer = null;
        this.autoSaveDelay = 1000; // 1 second debounce
        this.websocket = null;
        this.processingId = null;
        
        // Tabulator instance
        this.table = null;
        
        // API endpoints
        this.apiBase = '/api';
        this.endpoints = {
            upload: `${this.apiBase}/upload`,
            questions: `${this.apiBase}/questions`,
            updateQuestion: (id) => `${this.apiBase}/questions/${id}`,
            bulkOperations: `${this.apiBase}/questions/bulk`,
            saveApproved: `${this.apiBase}/questions/save`,
            export: `${this.apiBase}/export`,
            stats: `${this.apiBase}/stats`,
            websocket: `ws://${window.location.host}/api/ws/processing`
        };
        
        // Initialize the application
        this.init();
    }
    
    /**
     * Initialize the application
     */
    async init() {
        console.log('Initializing PDF Question Extractor App...');
        
        try {
            this.setupEventListeners();
            this.initializeTable();
            this.connectWebSocket();
            await this.loadStatistics();
            await this.loadQuestions();
            await this.loadFilterOptions();
            
            console.log('Application initialized successfully');
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showToast('Error', 'Failed to initialize application', 'error');
        }
    }
    
    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // File upload events
        this.setupFileUploadEvents();
        
        // Control panel events
        this.setupControlEvents();
        
        // Pagination events
        this.setupPaginationEvents();
        
        // Modal events
        this.setupModalEvents();
        
        // Search and filter events
        this.setupFilterEvents();
    }
    
    /**
     * Setup file upload event listeners
     */
    setupFileUploadEvents() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const selectFilesBtn = document.getElementById('selectFilesBtn');
        
        // Drag and drop events
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            const files = Array.from(e.dataTransfer.files).filter(file => 
                file.type === 'application/pdf'
            );
            
            if (files.length > 0) {
                this.handleFileUpload(files);
            } else {
                this.showToast('Warning', 'Please select only PDF files', 'warning');
            }
        });
        
        // Click to select files
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        selectFilesBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            if (files.length > 0) {
                this.handleFileUpload(files);
            }
        });
    }
    
    /**
     * Setup control panel event listeners
     */
    setupControlEvents() {
        // Bulk operations
        document.getElementById('bulkApproveBtn').addEventListener('click', () => {
            this.handleBulkOperation('approve');
        });
        
        document.getElementById('bulkRejectBtn').addEventListener('click', () => {
            this.handleBulkOperation('reject');
        });
        
        document.getElementById('bulkDeleteBtn').addEventListener('click', () => {
            this.handleBulkOperation('delete');
        });
        
        // Save approved questions
        document.getElementById('saveApprovedBtn').addEventListener('click', () => {
            this.handleSaveApproved();
        });
        
        // Export functionality
        const exportBtn = document.getElementById('exportBtn');
        const exportMenu = document.getElementById('exportMenu');
        
        exportBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            exportMenu.classList.toggle('show');
        });
        
        // Close export menu when clicking outside
        document.addEventListener('click', () => {
            exportMenu.classList.remove('show');
        });
        
        // Export format selection
        exportMenu.addEventListener('click', (e) => {
            if (e.target.classList.contains('dropdown-item')) {
                e.preventDefault();
                const format = e.target.dataset.format;
                this.handleExport(format);
                exportMenu.classList.remove('show');
            }
        });
    }
    
    /**
     * Setup pagination event listeners
     */
    setupPaginationEvents() {
        document.getElementById('prevPageBtn').addEventListener('click', () => {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.loadQuestions();
            }
        });
        
        document.getElementById('nextPageBtn').addEventListener('click', () => {
            const totalPages = Math.ceil(this.totalQuestions / this.perPage);
            if (this.currentPage < totalPages) {
                this.currentPage++;
                this.loadQuestions();
            }
        });
    }
    
    /**
     * Setup modal event listeners
     */
    setupModalEvents() {
        const modal = document.getElementById('confirmModal');
        const closeBtn = document.getElementById('confirmClose');
        const cancelBtn = document.getElementById('confirmCancel');
        
        closeBtn.addEventListener('click', () => {
            this.hideModal();
        });
        
        cancelBtn.addEventListener('click', () => {
            this.hideModal();
        });
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideModal();
            }
        });
    }
    
    /**
     * Setup filter and search event listeners
     */
    setupFilterEvents() {
        const statusFilter = document.getElementById('statusFilter');
        const yearFilter = document.getElementById('yearFilter');
        const levelFilter = document.getElementById('levelFilter');
        const searchInput = document.getElementById('searchInput');
        
        // Filter change events
        statusFilter.addEventListener('change', () => {
            this.currentPage = 1;
            this.loadQuestions();
        });
        
        yearFilter.addEventListener('change', () => {
            this.currentPage = 1;
            this.loadQuestions();
        });
        
        levelFilter.addEventListener('change', () => {
            this.currentPage = 1;
            this.loadQuestions();
        });
        
        // Search input with debounce
        let searchTimer = null;
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimer);
            searchTimer = setTimeout(() => {
                this.currentPage = 1;
                this.loadQuestions();
            }, 500);
        });
    }
    
    /**
     * Initialize Tabulator data grid
     */
    initializeTable() {
        const tableElement = document.getElementById('questionsTable');
        
        this.table = new Tabulator(tableElement, {
            layout: 'fitColumns',
            pagination: false, // We handle pagination manually
            selectable: true,
            responsiveLayout: 'collapse',
            movableColumns: true,
            resizableColumns: true,
            tooltips: true,
            history: true,
            
            columns: [
                {
                    formatter: 'rowSelection',
                    titleFormatter: 'rowSelection',
                    hozAlign: 'center',
                    headerSort: false,
                    cellClick: (_, cell) => {
                        cell.getRow().toggleSelect();
                    },
                    width: 40
                },
                {
                    title: 'ID',
                    field: 'id',
                    width: 80,
                    sorter: 'number',
                    hozAlign: 'center'
                },
                {
                    title: 'Status',
                    field: 'status',
                    width: 100,
                    formatter: this.statusFormatter,
                    editor: 'select',
                    editorParams: {
                        values: {
                            'pending': 'Pending',
                            'approved': 'Approved',
                            'rejected': 'Rejected'
                        }
                    },
                    cellEdited: (cell) => {
                        this.handleCellEdit(cell);
                    }
                },
                {
                    title: 'Question #',
                    field: 'question_number',
                    width: 100,
                    editor: 'input',
                    cellEdited: (cell) => {
                        this.handleCellEdit(cell);
                    }
                },
                {
                    title: 'Marks',
                    field: 'marks',
                    width: 80,
                    editor: 'number',
                    cellEdited: (cell) => {
                        this.handleCellEdit(cell);
                    }
                },
                {
                    title: 'Year',
                    field: 'year',
                    width: 80,
                    editor: 'input',
                    cellEdited: (cell) => {
                        this.handleCellEdit(cell);
                    }
                },
                {
                    title: 'Level',
                    field: 'level',
                    width: 100,
                    editor: 'input',
                    cellEdited: (cell) => {
                        this.handleCellEdit(cell);
                    }
                },
                {
                    title: 'Type',
                    field: 'question_type',
                    width: 120,
                    editor: 'input',
                    cellEdited: (cell) => {
                        this.handleCellEdit(cell);
                    }
                },
                {
                    title: 'Topics',
                    field: 'topics',
                    width: 150,
                    formatter: (cell) => {
                        const topics = cell.getValue();
                        return Array.isArray(topics) ? topics.join(', ') : (topics || '');
                    },
                    editor: 'input',
                    cellEdited: (cell) => {
                        this.handleCellEdit(cell);
                    }
                },
                {
                    title: 'Question Text',
                    field: 'question_text',
                    minWidth: 300,
                    formatter: 'textarea',
                    editor: 'textarea',
                    cellEdited: (cell) => {
                        this.handleCellEdit(cell);
                    }
                },
                {
                    title: 'Source PDF',
                    field: 'source_pdf',
                    width: 200,
                    formatter: (cell) => {
                        const filename = cell.getValue();
                        return filename ? filename.split('/').pop() : '';
                    }
                },
                {
                    title: 'Modified',
                    field: 'modified',
                    width: 80,
                    formatter: 'tickCross',
                    hozAlign: 'center'
                }
            ],
            
            // Row selection events
            rowSelectionChanged: (_, rows) => {
                this.selectedRows = rows;
                this.updateBulkActionButtons();
            },
            
            // Empty state
            placeholder: 'No questions available. Upload PDF files to get started.',
            
            // Loading state
            ajaxLoader: false,
            ajaxLoaderLoading: '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i></div>'
        });
    }
    
    /**
     * Status formatter for table cells
     */
    statusFormatter(cell) {
        const status = cell.getValue();
        const statusClass = `status-${status}`;
        return `<span class="status-badge ${statusClass}">${status}</span>`;
    }
    
    /**
     * Handle cell edit events with auto-save
     */
    handleCellEdit(cell) {
        const row = cell.getRow();
        const data = row.getData();
        
        // Mark as modified
        data.modified = true;
        row.update({ modified: true });
        
        // Trigger auto-save with debounce
        this.scheduleAutoSave(data.id, data);
    }
    
    /**
     * Schedule auto-save with debounce
     */
    scheduleAutoSave(questionId, data) {
        // Clear existing timer
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
        }
        
        // Show saving indicator
        this.showAutoSaveIndicator('saving');
        
        // Set new timer
        this.autoSaveTimer = setTimeout(async () => {
            try {
                await this.saveQuestion(questionId, data);
                this.showAutoSaveIndicator('saved');
            } catch (error) {
                console.error('Auto-save failed:', error);
                this.showAutoSaveIndicator('error');
            }
        }, this.autoSaveDelay);
    }
    
    /**
     * Show auto-save indicator
     */
    showAutoSaveIndicator(status) {
        let indicator = document.querySelector('.auto-save-indicator');
        
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'auto-save-indicator';
            document.body.appendChild(indicator);
        }
        
        // Remove existing classes
        indicator.classList.remove('saving', 'error');
        
        // Set content and class based on status
        switch (status) {
            case 'saving':
                indicator.textContent = 'Saving...';
                indicator.classList.add('saving');
                break;
            case 'saved':
                indicator.textContent = 'Saved';
                break;
            case 'error':
                indicator.textContent = 'Save failed';
                indicator.classList.add('error');
                break;
        }
        
        // Show indicator
        indicator.classList.add('show');
        
        // Hide after delay (except for saving state)
        if (status !== 'saving') {
            setTimeout(() => {
                indicator.classList.remove('show');
            }, 2000);
        }
    }
    
    /**
     * Connect to WebSocket for real-time updates
     */
    connectWebSocket() {
        try {
            this.websocket = new WebSocket(this.endpoints.websocket);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.showToast('Connected', 'Real-time updates enabled', 'success');
            };
            
            this.websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                // Attempt reconnection after 5 seconds
                setTimeout(() => {
                    this.connectWebSocket();
                }, 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }
    
    /**
     * Handle WebSocket messages
     */
    handleWebSocketMessage(message) {
        console.log('WebSocket message:', message);
        
        switch (message.type) {
            case 'connection_established':
                console.log('WebSocket connection established');
                break;
                
            case 'processing_started':
                console.log('Processing started:', message.data);
                this.showToast('Processing Started', 'Your PDF is being processed...', 'info');
                this.updateProcessingProgress({ progress_percentage: 0, current_step: 'Starting...' });
                break;
                
            case 'processing_progress':
            case 'progress':
                this.updateProcessingProgress(message.data);
                break;
                
            case 'processing_complete':
                this.handleProcessingComplete(message.data);
                break;
                
            case 'processing_error':
            case 'error':
                this.handleProcessingError(message.data);
                break;
                
            case 'question_extracted':
                this.handleQuestionExtracted(message.data);
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    /**
     * Update processing progress
     */
    updateProcessingProgress(data) {
        const progressBar = document.getElementById('uploadProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const progressPercent = document.getElementById('progressPercent');
        const progressStages = document.getElementById('progressStages');
        
        if (progressBar && progressFill && progressText && progressPercent) {
            progressBar.style.display = 'block';
            progressFill.style.width = `${data.progress_percentage || 0}%`;
            progressText.textContent = data.current_step || 'Processing...';
            progressPercent.textContent = `${Math.round(data.progress_percentage || 0)}%`;
            
            // Show progress stages
            if (progressStages) {
                progressStages.style.display = 'flex';
                
                // Update stage indicators based on current step
                const stages = {
                    'upload': ['Uploading', 'Upload'],
                    'ocr': ['OCR Processing', 'OCR', 'Reading'],
                    'extract': ['Extracting', 'Extract', 'Questions'],
                    'store': ['Storing', 'Store', 'Database', 'Saving']
                };
                
                // Reset all stages
                document.querySelectorAll('.progress-stage').forEach(stage => {
                    stage.classList.remove('active', 'completed');
                });
                
                // Find current stage
                const currentStep = (data.current_step || '').toLowerCase();
                let stageFound = false;
                let completedStages = [];
                
                for (const [stageId, keywords] of Object.entries(stages)) {
                    const stageElement = document.getElementById(`stage-${stageId}`);
                    if (stageElement) {
                        if (keywords.some(keyword => currentStep.includes(keyword.toLowerCase()))) {
                            stageElement.classList.add('active');
                            stageFound = true;
                        } else if (!stageFound) {
                            stageElement.classList.add('completed');
                            completedStages.push(stageId);
                        }
                    }
                }
                
                // Update progress based on stages
                if (!data.progress_percentage && completedStages.length > 0) {
                    const totalStages = 4;
                    const percentage = (completedStages.length / totalStages) * 100;
                    progressFill.style.width = `${percentage}%`;
                    progressPercent.textContent = `${Math.round(percentage)}%`;
                }
            }
        }
    }
    
    /**
     * Handle processing completion
     */
    async handleProcessingComplete(data) {
        // Hide progress indicators after a delay
        const progressBar = document.getElementById('uploadProgress');
        const progressStages = document.getElementById('progressStages');
        
        // Mark all stages as completed
        if (progressStages) {
            document.querySelectorAll('.progress-stage').forEach(stage => {
                stage.classList.remove('active');
                stage.classList.add('completed');
            });
        }
        
        // Hide after showing completion state
        if (progressBar) {
            setTimeout(() => {
                progressBar.style.display = 'none';
                if (progressStages) {
                    progressStages.style.display = 'none';
                }
            }, 2000);
        }
        
        this.showToast('Success', `Processing complete! ${data.questions_extracted || 0} questions extracted.`, 'success');
        
        // Reload questions and statistics
        await this.loadQuestions();
        await this.loadStatistics();
    }
    
    /**
     * Handle processing error
     */
    handleProcessingError(data) {
        const progressBar = document.getElementById('uploadProgress');
        const progressStages = document.getElementById('progressStages');
        
        if (progressBar) {
            progressBar.style.display = 'none';
        }
        if (progressStages) {
            progressStages.style.display = 'none';
        }
        
        this.showToast('Error', data.error || 'Processing failed', 'error');
    }
    
    /**
     * Handle question extracted event
     */
    handleQuestionExtracted(data) {
        // Could update UI with real-time question count
        console.log('Question extracted:', data);
    }
    
    /**
     * Handle file upload
     */
    async handleFileUpload(files) {
        if (!files || files.length === 0) {
            this.showToast('Warning', 'No files selected', 'warning');
            return;
        }
        
        try {
            const formData = new FormData();
            
            // Add files
            files.forEach(file => {
                formData.append('files', file);
            });
            
            // Add options
            const storeToDb = document.getElementById('storeToDb').checked;
            const generateEmbeddings = document.getElementById('generateEmbeddings').checked;
            const maxConcurrent = document.getElementById('maxConcurrent').value;
            
            // Append form fields
            formData.append('store_to_db', storeToDb);
            formData.append('generate_embeddings', generateEmbeddings);
            formData.append('max_concurrent', parseInt(maxConcurrent));
            
            // Show loading
            this.showLoading(true);
            
            const response = await fetch(this.endpoints.upload, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.processingId = result.processing_id;
                this.showToast('Success', result.message, 'success');
                
                // Start progress monitoring if WebSocket is connected
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(JSON.stringify({
                        type: 'start_processing',
                        processing_id: this.processingId
                    }));
                }
            } else {
                throw new Error(result.detail || 'Upload failed');
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showToast('Error', error.message || 'Upload failed', 'error');
        } finally {
            this.showLoading(false);
            
            // Clear file input
            document.getElementById('fileInput').value = '';
        }
    }
    
    /**
     * Load questions from API
     */
    async loadQuestions() {
        try {
            // Build query parameters
            const params = new URLSearchParams({
                page: this.currentPage,
                per_page: this.perPage,
                table: 'extracted' // Always load from extracted table for review
            });
            
            // Add filters
            const statusFilter = document.getElementById('statusFilter').value;
            const yearFilter = document.getElementById('yearFilter').value;
            const levelFilter = document.getElementById('levelFilter').value;
            const searchInput = document.getElementById('searchInput').value;
            
            if (statusFilter) params.append('status_filter', statusFilter);
            if (yearFilter) params.append('year_filter', yearFilter);
            if (levelFilter) params.append('level_filter', levelFilter);
            if (searchInput) params.append('search', searchInput);
            
            const response = await fetch(`${this.endpoints.questions}?${params}`);
            const data = await response.json();
            
            if (response.ok) {
                // Update table data
                this.table.setData(data.questions || []);
                
                // Update pagination info
                this.totalQuestions = data.total || 0;
                this.updatePaginationUI(data);
                
            } else {
                throw new Error(data.detail || 'Failed to load questions');
            }
            
        } catch (error) {
            console.error('Error loading questions:', error);
            this.showToast('Error', 'Failed to load questions', 'error');
        }
    }
    
    /**
     * Update pagination UI
     */
    updatePaginationUI(data) {
        const paginationInfo = document.getElementById('paginationInfo');
        const pageInfo = document.getElementById('pageInfo');
        const prevBtn = document.getElementById('prevPageBtn');
        const nextBtn = document.getElementById('nextPageBtn');
        
        // Update info text
        const start = data.total === 0 ? 0 : ((data.page - 1) * data.per_page) + 1;
        const end = Math.min(data.page * data.per_page, data.total);
        paginationInfo.textContent = `Showing ${start}-${end} of ${data.total} questions`;
        
        // Update page info
        pageInfo.textContent = `Page ${data.page} of ${data.total_pages}`;
        
        // Update button states
        prevBtn.disabled = !data.has_prev;
        nextBtn.disabled = !data.has_next;
    }
    
    /**
     * Load filter options from API
     */
    async loadFilterOptions() {
        try {
            // For now, we'll populate with static options
            // In a full implementation, you might fetch these from the API
            
            const yearFilter = document.getElementById('yearFilter');
            const levelFilter = document.getElementById('levelFilter');
            
            // Add common years (you could fetch these from stats API)
            const currentYear = new Date().getFullYear();
            for (let year = currentYear; year >= currentYear - 10; year--) {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                yearFilter.appendChild(option);
            }
            
            // Add common levels
            const levels = ['Primary', 'Secondary', 'A-Level', 'University'];
            levels.forEach(level => {
                const option = document.createElement('option');
                option.value = level;
                option.textContent = level;
                levelFilter.appendChild(option);
            });
            
        } catch (error) {
            console.error('Error loading filter options:', error);
        }
    }
    
    /**
     * Load system statistics
     */
    async loadStatistics() {
        try {
            const response = await fetch(this.endpoints.stats);
            const data = await response.json();
            
            if (response.ok) {
                // Update stats display
                document.getElementById('totalExtracted').textContent = data.total_extracted_questions || 0;
                document.getElementById('totalApproved').textContent = data.total_approved_questions || 0;
                document.getElementById('totalPermanent').textContent = data.total_permanent_questions || 0;
            }
            
        } catch (error) {
            console.error('Error loading statistics:', error);
        }
    }
    
    /**
     * Save a single question
     */
    async saveQuestion(questionId, data) {
        const response = await fetch(this.endpoints.updateQuestion(questionId), {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Save failed');
        }
        
        return await response.json();
    }
    
    /**
     * Update bulk action button states
     */
    updateBulkActionButtons() {
        const hasSelection = this.selectedRows.length > 0;
        
        document.getElementById('bulkApproveBtn').disabled = !hasSelection;
        document.getElementById('bulkRejectBtn').disabled = !hasSelection;
        document.getElementById('bulkDeleteBtn').disabled = !hasSelection;
    }
    
    /**
     * Handle bulk operations
     */
    async handleBulkOperation(operation) {
        if (this.selectedRows.length === 0) {
            this.showToast('Warning', 'No questions selected', 'warning');
            return;
        }
        
        const questionIds = this.selectedRows.map(row => row.getData().id);
        let operationEnum, confirmMessage;
        
        switch (operation) {
            case 'approve':
                operationEnum = 'APPROVE';
                confirmMessage = `Approve ${questionIds.length} selected questions?`;
                break;
            case 'reject':
                operationEnum = 'REJECT';
                confirmMessage = `Reject ${questionIds.length} selected questions?`;
                break;
            case 'delete':
                operationEnum = 'DELETE';
                confirmMessage = `Delete ${questionIds.length} selected questions? This action cannot be undone.`;
                break;
            default:
                return;
        }
        
        // Show confirmation modal
        const confirmed = await this.showConfirmation('Bulk Operation', confirmMessage);
        if (!confirmed) return;
        
        try {
            this.showLoading(true);
            
            const response = await fetch(this.endpoints.bulkOperations, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question_ids: questionIds,
                    operation: operationEnum
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showToast('Success', result.message, 'success');
                
                // Reload questions and clear selection
                await this.loadQuestions();
                await this.loadStatistics();
                this.table.deselectRow();
                
            } else {
                throw new Error(result.detail || 'Bulk operation failed');
            }
            
        } catch (error) {
            console.error('Bulk operation error:', error);
            this.showToast('Error', error.message || 'Bulk operation failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Handle save approved questions
     */
    async handleSaveApproved() {
        const confirmed = await this.showConfirmation(
            'Save Approved Questions',
            'Save all approved questions to permanent storage and clear from review table?'
        );
        
        if (!confirmed) return;
        
        try {
            this.showLoading(true);
            
            const response = await fetch(this.endpoints.saveApproved, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    clear_extracted: true
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showToast('Success', result.message, 'success');
                
                // Reload questions and statistics
                await this.loadQuestions();
                await this.loadStatistics();
                
            } else {
                throw new Error(result.detail || 'Save operation failed');
            }
            
        } catch (error) {
            console.error('Save approved error:', error);
            this.showToast('Error', error.message || 'Save operation failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Handle export functionality
     */
    async handleExport(format) {
        try {
            this.showLoading(true);
            
            // Build export parameters
            const params = new URLSearchParams({
                format: format,
                include_metadata: 'true'
            });
            
            // Add current filters
            const statusFilter = document.getElementById('statusFilter').value;
            const yearFilter = document.getElementById('yearFilter').value;
            const levelFilter = document.getElementById('levelFilter').value;
            
            if (statusFilter) params.append('status_filter', statusFilter);
            if (yearFilter) params.append('year_filter', yearFilter);
            if (levelFilter) params.append('level_filter', levelFilter);
            
            const response = await fetch(`${this.endpoints.export}?${params}`);
            const result = await response.json();
            
            if (result.success) {
                // Trigger download
                const downloadUrl = result.download_url;
                const link = document.createElement('a');
                link.href = downloadUrl;
                link.download = result.filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                this.showToast('Success', `Exported ${result.record_count} questions as ${format.toUpperCase()}`, 'success');
                
            } else {
                throw new Error(result.detail || 'Export failed');
            }
            
        } catch (error) {
            console.error('Export error:', error);
            this.showToast('Error', error.message || 'Export failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Show confirmation modal
     */
    showConfirmation(title, message) {
        return new Promise((resolve) => {
            const modal = document.getElementById('confirmModal');
            const titleElement = document.getElementById('confirmTitle');
            const messageElement = document.getElementById('confirmMessage');
            const okBtn = document.getElementById('confirmOk');
            
            titleElement.textContent = title;
            messageElement.textContent = message;
            modal.style.display = 'flex';
            
            // Handle confirm
            const handleConfirm = () => {
                cleanup();
                resolve(true);
            };
            
            // Cleanup function
            const cleanup = () => {
                modal.style.display = 'none';
                okBtn.removeEventListener('click', handleConfirm);
            };
            
            // Add event listeners
            okBtn.addEventListener('click', handleConfirm);
        });
    }
    
    /**
     * Hide modal
     */
    hideModal() {
        const modal = document.getElementById('confirmModal');
        modal.style.display = 'none';
    }
    
    /**
     * Show loading overlay
     */
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = show ? 'flex' : 'none';
    }
    
    /**
     * Show toast notification
     */
    showToast(title, message, type = 'info', duration = 5000) {
        const container = document.getElementById('toastContainer');
        
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        // Create icon based on type
        let icon;
        switch (type) {
            case 'success':
                icon = 'fas fa-check-circle';
                break;
            case 'error':
                icon = 'fas fa-exclamation-circle';
                break;
            case 'warning':
                icon = 'fas fa-exclamation-triangle';
                break;
            case 'info':
            default:
                icon = 'fas fa-info-circle';
        }
        
        toast.innerHTML = `
            <div class="toast-icon">
                <i class="${icon}"></i>
            </div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">&times;</button>
        `;
        
        // Add close functionality
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            this.removeToast(toast);
        });
        
        // Add to container
        container.appendChild(toast);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (toast.parentNode) {
                this.removeToast(toast);
            }
        }, duration);
    }
    
    /**
     * Remove toast notification
     */
    removeToast(toast) {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PDFQuestionExtractorApp();
});

// Export for module use if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PDFQuestionExtractorApp;
}