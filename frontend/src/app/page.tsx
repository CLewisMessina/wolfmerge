// frontend/src/app/page.tsx

"use client"

import { useState } from 'react'
import { FileUpload } from '@/components/file-upload'
import { DemoButton } from '@/components/demo-button'
import { SummaryDisplay } from '@/components/summary-display'
import { Card, CardContent } from '@/components/ui/card'
import { Brain, Shield, ArrowLeft, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function Home() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState<string>("")

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) return

    setIsAnalyzing(true)
    setError("")
    
    try {
      const formData = new FormData()
      selectedFiles.forEach(file => {
        formData.append('files', file)
      })

      const response = await fetch('http://localhost:8000/api/summarize', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Analysis failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (error: any) {
      console.error('Error:', error)
      setError(error.message || 'Analysis failed. Please try again.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleLoadDemo = () => {
    // Create demo files with realistic business content
    const demoFiles = [
      { 
        name: 'meeting_notes_jan.txt', 
        content: `Weekly Team Meeting - January 15, 2025

Attendees: Sarah, Mike, Jennifer, Alex
Duration: 1 hour

Key Discussion Points:
- Q1 goals and objectives review
- Budget allocation discussion - proposed $50,000 for marketing initiatives
- New hiring plans for development team
- Client feedback from December projects

Action Items:
- Sarah: Finalize marketing strategy document by January 31st
- Mike: Prepare hiring timeline and job descriptions
- Jennifer: Compile client feedback report
- Alex: Review Q1 budget proposals

Next Meeting: January 22, 2025`
      },
      { 
        name: 'project_proposal_v2.txt', 
        content: `Project Alpha - Development Proposal (Version 2)

Executive Summary:
Project Alpha represents a strategic initiative to develop our next-generation product offering. This proposal outlines the resource requirements, timeline, and expected deliverables.

Budget Request: $45,000
- Development: $30,000
- Research & Testing: $10,000  
- Marketing & Launch: $5,000

Timeline: 3 months (February - April 2025)

Key Deliverables:
1. Market research and competitive analysis
2. Product prototype development
3. User testing and feedback integration
4. Launch strategy documentation

Expected ROI: 150% within first year
Risk Assessment: Medium-low risk with high potential return

Approval Required: Management team sign-off by January 30th`
      },
      { 
        name: 'budget_overview.txt', 
        content: `Q1 2025 Budget Overview

Department Allocations:
- Marketing: $20,000
- Development: $30,000
- Operations: $15,000
- Research: $10,000
- Miscellaneous: $5,000

Total Q1 Budget: $80,000

Notes:
- Marketing budget reduced from initial $25,000 request
- Development increased to accommodate new projects
- Operations includes new software licenses
- Research budget allocated for market analysis

Important: Current project proposals requesting different amounts need reconciliation with actual budget allocations. Discrepancies noted in Project Alpha proposal ($45k vs $40k available).

Review Date: February 1, 2025
Approval Status: Pending final department reviews`
      }
    ]

    const files = demoFiles.map(demo => {
      const blob = new Blob([demo.content], { type: 'text/plain' })
      return new File([blob], demo.name, { type: 'text/plain' })
    })

    setSelectedFiles(files)
    setError("")
  }

  const handleReset = () => {
    setSelectedFiles([])
    setResult(null)
    setError("")
  }

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-600 mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">WolfMerge</h1>
                <p className="text-sm text-gray-600">Document Intelligence Platform</p>
              </div>
              <span className="ml-4 bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                Day 1 Alpha
              </span>
            </div>
            {result && (
              <Button 
                onClick={handleReset}
                variant="outline"
                className="flex items-center"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                New Analysis
              </Button>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-8">
        {!result ? (
          <>
            {/* Welcome Section */}
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Transform Document Chaos into Clarity
              </h2>
              <p className="text-xl text-gray-600 mb-6 max-w-2xl mx-auto">
                Upload your scattered documents and get instant, intelligent summaries that reveal themes, conflicts, and key insights
              </p>
              
              {/* Privacy Badge */}
              <div className="inline-flex items-center bg-green-50 border border-green-200 rounded-full px-4 py-2 mb-8">
                <Shield className="h-4 w-4 text-green-600 mr-2" />
                <span className="text-sm text-green-800">
                  🔒 Files processed securely, never stored
                </span>
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <Card className="mb-6 border-red-200 bg-red-50">
                <CardContent className="p-4">
                  <div className="flex items-center text-red-800">
                    <AlertTriangle className="h-4 w-4 mr-2" />
                    <span className="text-sm">{error}</span>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Upload Section */}
            <Card className="mb-6">
              <CardContent className="p-6">
                <FileUpload
                  onFilesSelected={setSelectedFiles}
                  onAnalyze={handleAnalyze}
                  selectedFiles={selectedFiles}
                  isAnalyzing={isAnalyzing}
                />
              </CardContent>
            </Card>

            {/* Demo Section */}
            <DemoButton 
              onLoadDemo={handleLoadDemo}
              isDisabled={isAnalyzing}
            />
          </>
        ) : (
          <SummaryDisplay result={result} />
        )}
      </div>
    </main>
  )
}