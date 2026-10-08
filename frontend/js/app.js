/**
 * Health Report Explainer — Frontend Application
 * =================================================
 * Handles:
 * - Navigation
 * - File upload
 * - Drag & drop
 * - Sample report
 * - API communication
 * - AI analysis
 * - Results rendering
 * - Language switching
 * - Mobile navigation
 * - UI state
 */


// ======================================================
// API BASE URL
// ======================================================

const API_BASE = window.location.origin;


// ======================================================
// APPLICATION STATE
// ======================================================

const state = {
    currentPage: 'landing',
    selectedFile: null,
    extractedText: '',
    currentAnalysis: null,
    currentLanguage: 'english',
    filename: '',
    fileType: '',
    isProcessing: false
};


// ======================================================
// LANGUAGE DISPLAY NAMES
// ======================================================

const LANGUAGE_NAMES = {
    english: 'English',
    telugu: 'Telugu (తెలుగు)',
    tamil: 'Tamil (தமிழ்)',
    hindi: 'Hindi (हिन्दी)',
    kannada: 'Kannada (ಕನ್ನಡ)',
    malayalam: 'Malayalam (മലയാളം)',
    bengali: 'Bengali (বাংলা)',
    marathi: 'Marathi (मराठी)',
    gujarati: 'Gujarati (ગુજરાતી)',
    punjabi: 'Punjabi (ਪੰਜਾਬੀ)',
    odia: 'Odia (ଓଡ଼ିଆ)'
};


// ======================================================
// FILE SETTINGS
// ======================================================

const ALLOWED_EXTENSIONS = [
    '.jpg',
    '.jpeg',
    '.png',
    '.pdf',
    '.doc',
    '.docx'
];

const MAX_FILE_SIZE = 10 * 1024 * 1024;


// ======================================================
// INITIALIZATION
// ======================================================

document.addEventListener('DOMContentLoaded', () => {

    console.log('Health Report Explainer starting...');

    try {
        initNavigation();
        initUpload();
        initAnalysis();
        initResultsLanguageSwitcher();
        initScrollEffects();
        initBgTracking();

        updateAnalyzeButton();

        console.log('Health Report Explainer initialized successfully.');

    } catch (error) {

        console.error('Initialization error:', error);

        showToast(
            'Some website features could not be initialized. Please refresh the page.',
            'error'
        );
    }

});


// ======================================================
// HELPER
// ======================================================

function getElement(id) {
    return document.getElementById(id);
}


// ======================================================
// NAVIGATION
// ======================================================

function initNavigation() {

    // --------------------------------------------------
    // Logo
    // --------------------------------------------------

    const logoLink = getElement('logoLink');

    if (logoLink) {

        logoLink.addEventListener('click', (event) => {

            event.preventDefault();

            closeMobileMenu();

            navigateTo('landing');

        });

    }


    // --------------------------------------------------
    // Navigation links
    // --------------------------------------------------

    document.querySelectorAll('[data-nav]').forEach((link) => {

        link.addEventListener('click', (event) => {

            const target = link.dataset.nav;

            closeMobileMenu();


            // HOME
            if (target === 'home') {

                event.preventDefault();

                navigateTo('landing');

                return;

            }


            // HOW IT WORKS / LANGUAGES / ABOUT
            if (
                target === 'how-it-works' ||
                target === 'languages' ||
                target === 'about'
            ) {

                event.preventDefault();

                navigateTo('landing');

                setTimeout(() => {

                    scrollToSection(target);

                }, 150);

            }

        });

    });


    // --------------------------------------------------
    // Explain My Report
    // --------------------------------------------------

    const navExplainBtn = getElement('navExplainBtn');

    if (navExplainBtn) {

        navExplainBtn.addEventListener('click', (event) => {

            event.preventDefault();

            closeMobileMenu();

            navigateTo('upload');

        });

    }


    // --------------------------------------------------
    // Hero Upload Report
    // --------------------------------------------------

    const heroUploadBtn = getElement('heroUploadBtn');

    if (heroUploadBtn) {

        heroUploadBtn.addEventListener('click', (event) => {

            event.preventDefault();

            navigateTo('upload');

        });

    }


    // --------------------------------------------------
    // Try It Now
    // --------------------------------------------------

    const stepsUploadBtn = getElement('stepsUploadBtn');

    if (stepsUploadBtn) {

        stepsUploadBtn.addEventListener('click', (event) => {

            event.preventDefault();

            navigateTo('upload');

        });

    }


    // --------------------------------------------------
    // Hero How It Works
    // --------------------------------------------------

    const heroHowBtn = getElement('heroHowBtn');

    if (heroHowBtn) {

        heroHowBtn.addEventListener('click', (event) => {

            event.preventDefault();

            closeMobileMenu();

            navigateTo('landing');

            setTimeout(() => {

                scrollToSection('how-it-works');

            }, 150);

        });

    }


    // --------------------------------------------------
    // Analyze Another Report
    // --------------------------------------------------

    const newReportBtn = getElement('newReportBtn');

    if (newReportBtn) {

        newReportBtn.addEventListener('click', (event) => {

            event.preventDefault();

            resetUploadState();

            navigateTo('upload');

        });

    }


    // --------------------------------------------------
    // Mobile menu
    // --------------------------------------------------

    const mobileMenuBtn = getElement('mobileMenuBtn');
    const navLinks = getElement('navLinks');

    if (mobileMenuBtn && navLinks) {

        mobileMenuBtn.addEventListener('click', (event) => {

            event.preventDefault();

            navLinks.classList.toggle('open');

        });

    }


    // --------------------------------------------------
    // Close mobile menu when a link is clicked
    // --------------------------------------------------

    document.querySelectorAll('.nav-links a').forEach((link) => {

        link.addEventListener('click', () => {

            closeMobileMenu();

        });

    });

}


// ======================================================
// CHANGE SPA PAGE
// ======================================================

function navigateTo(page) {

    console.log('Navigating to:', page);


    // Hide all pages
    document.querySelectorAll('.page').forEach((pageElement) => {

        pageElement.classList.remove('active');

    });


    // Find requested page
    const targetPage = getElement(`page-${page}`);

    if (!targetPage) {

        console.error(`Page not found: page-${page}`);

        return;

    }


    // Show requested page
    targetPage.classList.add('active');

    state.currentPage = page;


    // Update navigation
    updateActiveNavigation(page);


    // Scroll to top
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });

}


// ======================================================
// ACTIVE NAVIGATION
// ======================================================

function updateActiveNavigation(page) {

    document.querySelectorAll('[data-nav]').forEach((link) => {

        link.classList.remove('active');

    });


    if (page === 'landing') {

        const homeLink = document.querySelector('[data-nav="home"]');

        if (homeLink) {

            homeLink.classList.add('active');

        }

    }

}


// ======================================================
// SCROLL TO SECTION
// ======================================================

function scrollToSection(sectionId) {

    const section = getElement(sectionId);

    if (!section) {

        console.warn(`Section not found: ${sectionId}`);

        return;

    }


    section.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });

}


// ======================================================
// CLOSE MOBILE MENU
// ======================================================

function closeMobileMenu() {

    const navLinks = getElement('navLinks');

    if (navLinks) {

        navLinks.classList.remove('open');
        navLinks.classList.remove('show');

    }

}


// ======================================================
// SCROLL EFFECT
// ======================================================

function initScrollEffects() {

    const header = getElement('mainHeader');

    if (!header) {

        return;

    }


    window.addEventListener('scroll', () => {

        if (window.scrollY > 20) {

            header.classList.add('scrolled');

        } else {

            header.classList.remove('scrolled');

        }

    });

}


// ======================================================
// BACKGROUND MOUSE TRACKING
// ======================================================

function initBgTracking() {

    const bg = getElement('bgPattern');

    if (!bg) {

        return;

    }


    document.addEventListener('mousemove', (event) => {

        const x = (event.clientX / window.innerWidth) * 100;
        const y = (event.clientY / window.innerHeight) * 100;

        bg.style.setProperty('--mouse-x', `${x}%`);
        bg.style.setProperty('--mouse-y', `${y}%`);

    });

}


// ======================================================
// FILE UPLOAD INITIALIZATION
// ======================================================

function initUpload() {

    const uploadZone = getElement('uploadZone');
    const fileInput = getElement('fileInput');
    const browseBtn = getElement('browseBtn');
    const fileRemoveBtn = getElement('fileRemoveBtn');
    const sampleReportBtn = getElement('sampleReportBtn');


    if (!uploadZone || !fileInput) {

        console.error('Upload zone or file input not found.');

        return;

    }


    // --------------------------------------------------
    // Browse Files button
    // --------------------------------------------------

    if (browseBtn) {

        browseBtn.addEventListener('click', (event) => {

            event.preventDefault();

            // Prevent upload zone click from firing again
            event.stopPropagation();

            fileInput.click();

        });

    }


    // --------------------------------------------------
    // Upload zone click
    // --------------------------------------------------

    uploadZone.addEventListener('click', () => {

        fileInput.click();

    });


    // --------------------------------------------------
    // Keyboard support
    // --------------------------------------------------

    uploadZone.addEventListener('keydown', (event) => {

        if (
            event.key === 'Enter' ||
            event.key === ' '
        ) {

            event.preventDefault();

            fileInput.click();

        }

    });


    // --------------------------------------------------
    // File selected
    // --------------------------------------------------

    fileInput.addEventListener('change', (event) => {

        const files = event.target.files;

        if (files && files.length > 0) {

            handleFileSelect(files[0]);

        }

    });


    // --------------------------------------------------
    // Drag over
    // --------------------------------------------------

    uploadZone.addEventListener('dragover', (event) => {

        event.preventDefault();

        uploadZone.classList.add('drag-over');

    });


    // --------------------------------------------------
    // Drag leave
    // --------------------------------------------------

    uploadZone.addEventListener('dragleave', () => {

        uploadZone.classList.remove('drag-over');

    });


    // --------------------------------------------------
    // Drop
    // --------------------------------------------------

    uploadZone.addEventListener('drop', (event) => {

        event.preventDefault();

        uploadZone.classList.remove('drag-over');


        const files = event.dataTransfer.files;

        if (files && files.length > 0) {

            handleFileSelect(files[0]);

        }

    });


    // --------------------------------------------------
    // Remove selected file
    // --------------------------------------------------

    if (fileRemoveBtn) {

        fileRemoveBtn.addEventListener('click', (event) => {

            event.preventDefault();

            event.stopPropagation();

            resetUploadState();

        });

    }


    // --------------------------------------------------
    // Sample report
    // --------------------------------------------------

    if (sampleReportBtn) {

        sampleReportBtn.addEventListener('click', (event) => {

            event.preventDefault();

            event.stopPropagation();

            loadSampleReport();

        });

    }

}


// ======================================================
// HANDLE FILE SELECTION
// ======================================================

function handleFileSelect(file) {

    if (!file) {

        return;

    }


    // --------------------------------------------------
    // Extension
    // --------------------------------------------------

    const filename = file.name || '';

    const lastDot = filename.lastIndexOf('.');

    const ext = lastDot >= 0
        ? filename.substring(lastDot).toLowerCase()
        : '';


    // --------------------------------------------------
    // Validate extension
    // --------------------------------------------------

    if (!ALLOWED_EXTENSIONS.includes(ext)) {

        showToast(
            'This file type is not supported. Please upload JPG, JPEG, PNG, PDF, DOC, or DOCX.',
            'error'
        );

        return;

    }


    // --------------------------------------------------
    // Validate size
    // --------------------------------------------------

    if (file.size > MAX_FILE_SIZE) {

        showToast(
            `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed is 10 MB.`,
            'error'
        );

        return;

    }


    // --------------------------------------------------
    // Validate empty file
    // --------------------------------------------------

    if (file.size === 0) {

        showToast(
            'The uploaded file is empty. Please upload a valid report.',
            'error'
        );

        return;

    }


    // --------------------------------------------------
    // Save state
    // --------------------------------------------------

    state.selectedFile = file;
    state.extractedText = '';
    state.filename = file.name;


    // --------------------------------------------------
    // File information
    // --------------------------------------------------

    const fileInfo = getElement('fileInfo');
    const fileIcon = getElement('fileIcon');
    const fileName = getElement('fileName');
    const fileMeta = getElement('fileMeta');
    const uploadZone = getElement('uploadZone');


    if (fileInfo) {

        fileInfo.classList.remove('hidden');

    }


    // --------------------------------------------------
    // File icons
    // --------------------------------------------------

    const iconMap = {

        '.jpg': '🖼️',
        '.jpeg': '🖼️',
        '.png': '🖼️',
        '.pdf': '📕',
        '.doc': '📘',
        '.docx': '📘'

    };


    if (fileIcon) {

        fileIcon.textContent = iconMap[ext] || '📄';

    }


    if (fileName) {

        fileName.textContent = file.name;

    }


    // --------------------------------------------------
    // File type
    // --------------------------------------------------

    const typeMap = {

        '.jpg': 'JPEG Image',
        '.jpeg': 'JPEG Image',
        '.png': 'PNG Image',
        '.pdf': 'PDF Document',
        '.doc': 'Word Document',
        '.docx': 'Word Document'

    };


    state.fileType = typeMap[ext] || 'Document';


    if (fileMeta) {

        fileMeta.textContent =
            `${state.fileType} • ${formatFileSize(file.size)}`;

    }


    if (uploadZone) {

        uploadZone.classList.add('has-file');

    }


    // --------------------------------------------------
    // Enable Analyze button when consent is checked
    // --------------------------------------------------

    updateAnalyzeButton();


    showToast(
        `${file.name} selected successfully.`,
        'success'
    );

}


// ======================================================
// RESET UPLOAD STATE
// ======================================================

function resetUploadState() {

    state.selectedFile = null;
    state.extractedText = '';
    state.filename = '';
    state.fileType = '';
    state.currentAnalysis = null;
    state.isProcessing = false;


    const fileInput = getElement('fileInput');
    const fileInfo = getElement('fileInfo');
    const uploadZone = getElement('uploadZone');
    const consentCheckbox = getElement('consentCheckbox');


    if (fileInput) {

        fileInput.value = '';

    }


    if (fileInfo) {

        fileInfo.classList.add('hidden');

    }


    if (uploadZone) {

        uploadZone.classList.remove('has-file');
        uploadZone.classList.remove('drag-over');

    }


    if (consentCheckbox) {

        consentCheckbox.checked = false;

    }


    updateAnalyzeButton();

}


// ======================================================
// FORMAT FILE SIZE
// ======================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {

        return `${bytes} B`;

    }


    if (bytes < 1024 * 1024) {

        return `${(bytes / 1024).toFixed(1)} KB`;

    }


    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

}


// ======================================================
// SAMPLE REPORT
// ======================================================

async function loadSampleReport() {

    const sampleReportBtn = getElement('sampleReportBtn');

    try {

        if (sampleReportBtn) {

            sampleReportBtn.disabled = true;

        }


        showToast(
            'Loading sample medical report...',
            'success'
        );


        const response = await fetch(
            `${API_BASE}/api/sample-report`
        );


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );

        }


        const data = await response.json();


        if (!data.success || !data.extracted_text) {

            throw new Error(
                'Sample report data was not returned.'
            );

        }


        state.extractedText = data.extracted_text;
        state.filename = data.filename || 'sample_report.txt';
        state.fileType = data.file_type || 'Sample Report';

        // Special marker for sample report
        state.selectedFile = 'sample';


        // --------------------------------------------------
        // Display sample file
        // --------------------------------------------------

        const fileInfo = getElement('fileInfo');
        const fileIcon = getElement('fileIcon');
        const fileName = getElement('fileName');
        const fileMeta = getElement('fileMeta');
        const uploadZone = getElement('uploadZone');


        if (fileInfo) {

            fileInfo.classList.remove('hidden');

        }


        if (fileIcon) {

            fileIcon.textContent = '📋';

        }


        if (fileName) {

            fileName.textContent =
                'Sample Medical Report (Demo)';

        }


        if (fileMeta) {

            fileMeta.textContent =
                'Fictional CBC + Iron + Liver + Kidney + Sugar Report';

        }


        if (uploadZone) {

            uploadZone.classList.add('has-file');

        }


        updateAnalyzeButton();


        showToast(
            'Sample report loaded! Accept the disclaimer and click Analyze Report.',
            'success'
        );


    } catch (error) {

        console.error(
            'Sample report error:',
            error
        );


        showToast(
            'Unable to load sample report. Please make sure the backend is running.',
            'error'
        );


    } finally {

        if (sampleReportBtn) {

            sampleReportBtn.disabled = false;

        }

    }

}


// ======================================================
// ANALYSIS INITIALIZATION
// ======================================================

function initAnalysis() {

    const analyzeBtn = getElement('analyzeBtn');
    const consentCheckbox = getElement('consentCheckbox');


    if (consentCheckbox) {

        consentCheckbox.addEventListener(
            'change',
            updateAnalyzeButton
        );

    }


    if (analyzeBtn) {

        analyzeBtn.addEventListener(
            'click',
            (event) => {

                event.preventDefault();

                startAnalysis();

            }
        );

    }

}


// ======================================================
// UPDATE ANALYZE BUTTON
// ======================================================

function updateAnalyzeButton() {

    const analyzeBtn = getElement('analyzeBtn');
    const consentCheckbox = getElement('consentCheckbox');


    if (!analyzeBtn || !consentCheckbox) {

        return;

    }


    const hasFile =
        state.selectedFile !== null;


    const consent =
        consentCheckbox.checked;


    const canAnalyze =
        hasFile &&
        consent &&
        !state.isProcessing;


    analyzeBtn.disabled = !canAnalyze;


    if (canAnalyze) {

        analyzeBtn.classList.remove('disabled');

    } else {

        analyzeBtn.classList.add('disabled');

    }

}


// ======================================================
// START ANALYSIS
// ======================================================

async function startAnalysis() {

    // Prevent duplicate requests
    if (state.isProcessing) {

        return;

    }


    // Check file
    if (!state.selectedFile) {

        showToast(
            'Please select a medical report first.',
            'error'
        );

        return;

    }


    // Check consent
    const consentCheckbox =
        getElement('consentCheckbox');


    if (!consentCheckbox || !consentCheckbox.checked) {

        showToast(
            'Please accept the medical disclaimer before analyzing.',
            'error'
        );

        return;

    }


    // Get selected language
    const languageSelect =
        getElement('languageSelect');


    state.currentLanguage =
        languageSelect
            ? languageSelect.value
            : 'english';


    state.isProcessing = true;

    updateAnalyzeButton();


    // Go to processing screen
    navigateTo('processing');

    resetProcessingSteps();


    try {

        // --------------------------------------------------
        // Step 1: Upload
        // --------------------------------------------------

        updateProcessingStep(
            'step-upload',
            'active'
        );


        await delay(300);


        // --------------------------------------------------
        // Step 2: Read
        // --------------------------------------------------

        updateProcessingStep(
            'step-upload',
            'done'
        );

        updateProcessingStep(
            'step-read',
            'active'
        );


        await delay(300);


        // --------------------------------------------------
        // Step 3: Extract
        // --------------------------------------------------

        updateProcessingStep(
            'step-read',
            'done'
        );

        updateProcessingStep(
            'step-extract',
            'active'
        );


        // --------------------------------------------------
        // Call backend
        // --------------------------------------------------

        const analysis = await analyzeReport(
            state.extractedText || '',
            state.currentLanguage
        );


        state.currentAnalysis = analysis;


        updateProcessingStep(
            'step-extract',
            'done'
        );


        // --------------------------------------------------
        // AI Analysis step
        // --------------------------------------------------

        updateProcessingStep(
            'step-analyze',
            'active'
        );


        await delay(300);


        updateProcessingStep(
            'step-analyze',
            'done'
        );


        // --------------------------------------------------
        // Prepare explanation
        // --------------------------------------------------

        updateProcessingStep(
            'step-prepare',
            'active'
        );


        await delay(500);


        updateProcessingStep(
            'step-prepare',
            'done'
        );


        // --------------------------------------------------
        // Complete
        // --------------------------------------------------

        updateProcessingStep(
            'step-complete',
            'done'
        );


        await delay(500);


        // --------------------------------------------------
        // Render results
        // --------------------------------------------------

        renderResults(analysis);

        navigateTo('results');


    } catch (error) {

        console.error(
            'Analysis error:',
            error
        );


        showToast(
            error.message ||
            'Unable to analyze the report right now. Please try again.',
            'error'
        );


        navigateTo('upload');


    } finally {

        state.isProcessing = false;

        updateAnalyzeButton();

    }

}


// ======================================================
// ANALYZE REPORT
// ======================================================

async function analyzeReport(
    extractedText,
    language
) {

    const formData = new FormData();


    // --------------------------------------------------
    // Real uploaded file
    // --------------------------------------------------

    if (
        state.selectedFile &&
        state.selectedFile !== 'sample'
    ) {

        formData.append(
            'file',
            state.selectedFile
        );

    }


    // --------------------------------------------------
    // Extracted text
    // --------------------------------------------------

    if (extractedText) {

        formData.append(
            'extracted_text',
            extractedText
        );

    }


    // --------------------------------------------------
    // Language
    // --------------------------------------------------

    formData.append(
        'language',
        language
    );


    console.log(
        'Sending report for analysis...',
        {
            filename: state.filename,
            language: language,
            hasFile:
                state.selectedFile !== null
        }
    );


    const response = await fetch(
        `${API_BASE}/api/analyze`,
        {
            method: 'POST',
            body: formData
        }
    );


    // --------------------------------------------------
    // Handle API error
    // --------------------------------------------------

    if (!response.ok) {

        let errorMessage =
            'Unable to analyze the report right now.';


        try {

            const errorData =
                await response.json();


            errorMessage =
                errorData.detail ||
                errorData.message ||
                errorMessage;

        } catch {

            // Ignore JSON parsing failure

        }


        throw new Error(errorMessage);

    }


    // --------------------------------------------------
    // Parse result
    // --------------------------------------------------

    const data =
        await response.json();


    console.log(
        'Analysis response:',
        data
    );


    return {

        success:
            data.success !== false,

        language:
            data.language ||
            language,

        explanation:
            data.explanation ||
            data.summary ||
            '',

        report_name:
            data.report_name ||
            data.filename ||
            state.filename ||
            'medical_report'

    };

}


// ======================================================
// PROCESSING STEPS
// ======================================================

function resetProcessingSteps() {

    const steps = [
        'step-upload',
        'step-read',
        'step-extract',
        'step-analyze',
        'step-prepare',
        'step-complete'
    ];


    steps.forEach((id) => {

        const element = getElement(id);

        if (!element) {

            return;

        }


        element.className =
            'processing-step pending';


        const icon =
            element.querySelector('.step-icon');


        if (icon) {

            icon.textContent = '○';

        }

    });

}


// ======================================================
// UPDATE PROCESSING STEP
// ======================================================

function updateProcessingStep(
    id,
    status
) {

    const element =
        getElement(id);


    if (!element) {

        return;

    }


    element.className =
        `processing-step ${status}`;


    const icon =
        element.querySelector('.step-icon');


    if (!icon) {

        return;

    }


    if (status === 'done') {

        icon.textContent = '✓';

    } else if (status === 'active') {

        icon.textContent = '●';

    } else {

        icon.textContent = '○';

    }

}


// ======================================================
// RESULT LANGUAGE SWITCHER
// ======================================================

function initResultsLanguageSwitcher() {

    const select =
        getElement('resultLanguageSelect');


    if (!select) {

        return;

    }


    select.addEventListener(
        'change',
        async () => {

            const newLanguage =
                select.value;


            if (
                newLanguage ===
                state.currentLanguage
            ) {

                return;

            }


            if (!state.currentAnalysis) {

                showToast(
                    'There is no analysis available to translate.',
                    'error'
                );

                return;

            }


            const oldLanguage =
                state.currentLanguage;


            state.currentLanguage =
                newLanguage;


            const resultLanguage =
                getElement('resultLanguage');


            if (resultLanguage) {

                resultLanguage.textContent =
                    '🌐 ' +
                    (
                        LANGUAGE_NAMES[newLanguage] ||
                        newLanguage
                    );

            }


            select.disabled = true;


            showToast(
                `Generating explanation in ${LANGUAGE_NAMES[newLanguage] || newLanguage}...`,
                'success'
            );


            try {

                const response =
                    await fetch(
                        `${API_BASE}/api/translate`,
                        {
                            method: 'POST',

                            headers: {
                                'Content-Type':
                                    'application/json'
                            },

                            body: JSON.stringify({

                                analysis:
                                    state.currentAnalysis,

                                target_language:
                                    newLanguage,

                                extracted_text:
                                    state.extractedText

                            })

                        }
                    );


                if (!response.ok) {

                    let message =
                        'Translation failed.';


                    try {

                        const errorData =
                            await response.json();


                        message =
                            errorData.detail ||
                            errorData.message ||
                            message;

                    } catch {

                        // Ignore

                    }


                    throw new Error(message);

                }


                const data =
                    await response.json();


                state.currentAnalysis =
                    data.analysis ||
                    data;


                renderResults(
                    state.currentAnalysis
                );


                showToast(
                    `Explanation updated to ${LANGUAGE_NAMES[newLanguage] || newLanguage}`,
                    'success'
                );


            } catch (error) {

                console.error(
                    'Translation error:',
                    error
                );


                state.currentLanguage =
                    oldLanguage;


                showToast(
                    error.message ||
                    'Unable to translate. Please try again.',
                    'error'
                );


            } finally {

                select.disabled = false;

            }

        }
    );

}


// ======================================================
// RENDER RESULTS
// ======================================================

function renderResults(analysis) {

    const grid =
        getElement('resultsGrid');


    if (!grid) {

        console.error(
            'Results grid not found.'
        );

        return;

    }


    grid.innerHTML = '';


    // --------------------------------------------------
    // Normalize response
    // --------------------------------------------------

    let response = analysis;


    if (
        !response ||
        !response.explanation
    ) {

        response = {

            language:
                state.currentLanguage,

            explanation:
                response?.summary ||
                'No explanation available.',

            report_name:
                state.filename ||
                'medical_report'

        };

    }


    const reportName =
        response.report_name ||
        state.filename ||
        'medical_report';


    const language =
        response.language ||
        state.currentLanguage ||
        'english';


    // --------------------------------------------------
    // Header
    // --------------------------------------------------

    const resultFilename =
        getElement('resultFilename');


    if (resultFilename) {

        resultFilename.textContent =
            '📄 ' + reportName;

    }


    const resultLanguage =
        getElement('resultLanguage');


    if (resultLanguage) {

        resultLanguage.textContent =
            '🌐 ' +
            (
                LANGUAGE_NAMES[language] ||
                language
            );

    }


    const resultLanguageSelect =
        getElement('resultLanguageSelect');


    if (resultLanguageSelect) {

        resultLanguageSelect.value =
            language;

    }


    // --------------------------------------------------
    // Explanation card
    // --------------------------------------------------

    grid.appendChild(
        createCard(
            '🧾',
            'Your Report Explanation',
            formatExplanationCard(
                response.explanation ||
                'No explanation available.'
            )
        )
    );


    // --------------------------------------------------
    // Medical disclaimer
    // --------------------------------------------------

    grid.appendChild(
        createDisclaimerCard(
            'This explanation is for informational purposes only and is not a medical diagnosis or a substitute for professional medical advice.'
        )
    );


    // --------------------------------------------------
    // Animation
    // --------------------------------------------------

    grid
        .querySelectorAll('.result-card')
        .forEach((card, index) => {

            card.style.animationDelay =
                `${index * 0.08}s`;

            card.classList.add(
                'animate-in'
            );

        });

}


// ======================================================
// FORMAT EXPLANATION
// ======================================================

function formatExplanationCard(text) {

    const normalizedText =
        (text || '').trim();


    if (!normalizedText) {

        return `
            <p class="summary-text">
                No explanation was returned.
            </p>
        `;

    }


    const sections =
        normalizedText
            .split(/\n\s*\n/)
            .map(section => section.trim())
            .filter(Boolean);


    let html = '';


    sections.forEach((section) => {

        // Markdown heading
        const headingMatch =
            section.match(/^###\s*(.+)$/i);


        if (headingMatch) {

            html += `
                <h4>
                    ${escapeHtml(
                        headingMatch[1].trim()
                    )}
                </h4>
            `;


            const body =
                section
                    .replace(
                        /^###\s*.+$/i,
                        ''
                    )
                    .trim();


            if (body) {

                html += `
                    <p>
                        ${escapeHtml(body)
                            .replace(/\n/g, '<br>')}
                    </p>
                `;

            }


            return;

        }


        // Normal paragraph
        html += `
            <p>
                ${escapeHtml(section)
                    .replace(/\n/g, '<br>')}
            </p>
        `;

    });


    if (!html) {

        html = `
            <p>
                ${escapeHtml(normalizedText)
                    .replace(/\n/g, '<br>')}
            </p>
        `;

    }


    return `
        <div class="summary-text">
            ${html}
        </div>
    `;

}


// ======================================================
// CREATE RESULT CARD
// ======================================================

function createCard(
    icon,
    title,
    bodyHtml
) {

    const card =
        document.createElement('div');


    card.className =
        'result-card';


    card.innerHTML = `

        <div class="result-card-header">

            <span class="card-icon">
                ${icon}
            </span>

            <h3>
                ${escapeHtml(title)}
            </h3>

        </div>

        <div class="result-card-body">

            ${bodyHtml}

        </div>

    `;


    return card;

}


// ======================================================
// DISCLAIMER CARD
// ======================================================

function createDisclaimerCard(
    disclaimer
) {

    const card =
        document.createElement('div');


    card.className =
        'result-card disclaimer-card';


    card.innerHTML = `

        <div class="result-card-header">

            <span class="card-icon">
                ⚕️
            </span>

            <h3>
                Medical Disclaimer
            </h3>

        </div>

        <div class="result-card-body">

            <p class="disclaimer-text">

                ${escapeHtml(
                    disclaimer ||
                    'This tool provides AI-generated explanations for informational purposes only.'
                )}

            </p>

            <div class="disclaimer-emergency">

                🚑 If you are experiencing severe or emergency symptoms, seek immediate medical care.

            </div>

        </div>

    `;


    return card;

}


// ======================================================
// STATUS ICON
// ======================================================

function getStatusIcon(status) {

    const value =
        (status || '').toLowerCase();


    if (value === 'normal') {

        return '✅';

    }


    if (value === 'high') {

        return '🔺';

    }


    if (value === 'low') {

        return '🔻';

    }


    if (value === 'critical') {

        return '🚨';

    }


    return '•';

}


// ======================================================
// TOAST
// ======================================================

function showToast(
    message,
    type = 'error'
) {

    const container =
        getElement('toastContainer');


    if (!container) {

        console.log(message);

        return;

    }


    const toast =
        document.createElement('div');


    toast.className =
        `toast ${type}`;


    toast.innerHTML = `

        <span>
            ${
                type === 'error'
                    ? '❌'
                    : '✅'
            }
        </span>

        <span>
            ${escapeHtml(message)}
        </span>

        <button
            class="toast-close"
            aria-label="Close notification"
            type="button"
        >
            ✕
        </button>

    `;


    const closeButton =
        toast.querySelector(
            '.toast-close'
        );


    if (closeButton) {

        closeButton.addEventListener(
            'click',
            () => toast.remove()
        );

    }


    container.appendChild(toast);


    // Auto-remove after 6 seconds
    setTimeout(() => {

        if (!toast.parentNode) {

            return;

        }


        toast.style.opacity = '0';

        toast.style.transform =
            'translateX(100px)';

        toast.style.transition =
            'all 0.3s ease';


        setTimeout(() => {

            if (toast.parentNode) {

                toast.remove();

            }

        }, 300);

    }, 6000);

}


// ======================================================
// ESCAPE HTML
// ======================================================

function escapeHtml(text) {

    if (text === null || text === undefined) {

        return '';

    }


    const div =
        document.createElement('div');


    div.textContent =
        String(text);


    return div.innerHTML;

}


// ======================================================
// DELAY
// ======================================================

function delay(ms) {

    return new Promise(
        resolve => setTimeout(resolve, ms)
    );

}