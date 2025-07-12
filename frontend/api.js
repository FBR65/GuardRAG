// FastAPI Backend Integration
class GuardRAGAPI {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    // Helper für API-Aufrufe
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const requestOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error for ${endpoint}:`, error);
            throw error;
        }
    }

    // System-Endpoints
    async getHealth() {
        return await this.makeRequest('/health');
    }

    async getSystemStatus() {
        return await this.makeRequest('/system-status');
    }

    async getEnhancedSystemStatus() {
        return await this.makeRequest('/enhanced-system-status');
    }

    // COLPALI-Endpoints
    async getColpaliStats() {
        return await this.makeRequest('/colpali-stats');
    }

    async clearColpaliCache() {
        return await this.makeRequest('/colpali-clear-cache', {
            method: 'POST'
        });
    }

    // Datei-Upload
    async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);

        return await fetch(`${this.baseURL}/upload-document`, {
            method: 'POST',
            body: formData
        }).then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.detail || `Upload failed: ${response.statusText}`);
                });
            }
            return response.json();
        });
    }

    // RAG-Abfragen
    async ragQuery(query) {
        return await this.makeRequest('/rag-query', {
            method: 'POST',
            body: JSON.stringify({ query })
        });
    }

    async enhancedRagQuery(query, collectionName = 'default_collection') {
        return await this.makeRequest('/enhanced-rag-query', {
            method: 'POST',
            body: JSON.stringify({ 
                query,
                collection_name: collectionName
            })
        });
    }

    // Dokumentenmanagement
    async clearDocuments() {
        return await this.makeRequest('/clear-documents', {
            method: 'DELETE'
        });
    }

    // Sicherheits-Endpoints
    async getSecurityStats() {
        return await this.makeRequest('/security/stats');
    }

    async updateSecurityLevel(level) {
        return await this.makeRequest(`/security/update-level?level=${level}`, {
            method: 'POST'
        });
    }

    async updateCompetitors(competitors) {
        return await this.makeRequest('/update-competitors', {
            method: 'POST',
            body: JSON.stringify(competitors)
        });
    }

    async resetEnhancedStatistics() {
        return await this.makeRequest('/reset-enhanced-statistics', {
            method: 'POST'
        });
    }

    // PDF-Konvertierung
    async convertToPdf(inputFilepath, outputDirectory = './uploads', outputFilename = null) {
        return await this.makeRequest('/convert-to-pdf', {
            method: 'POST',
            body: JSON.stringify({
                input_filepath: inputFilepath,
                output_directory: outputDirectory,
                output_filename: outputFilename
            })
        });
    }
}

// Globale API-Instanz
const api = new GuardRAGAPI();
