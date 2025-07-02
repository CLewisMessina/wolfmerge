// frontend/src/components/summary-display.tsx

"use client"

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { FileText, Layers, Clock, TrendingUp, AlertTriangle } from 'lucide-react'

interface SummaryResult {
  individual_summaries: Array<{
    filename: string
    summary: string
    original_size: number
  }>
  unified_summary: string
  document_count: number
  total_size: number
  processing_time: string
}

interface SummaryDisplayProps {
  result: SummaryResult
}

export function SummaryDisplay({ result }: SummaryDisplayProps) {
  const totalSizeKB = (result.total_size / 1024).toFixed(1)
  
  return (
    <div className="space-y-6">
      {/* Success Banner */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <TrendingUp className="h-5 w-5 text-green-600" />
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-green-800">
              Analysis Complete!
            </h3>
            <p className="text-sm text-green-700">
              Successfully processed {result.document_count} documents ({totalSizeKB} KB total)
            </p>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <FileText className="h-6 w-6 mx-auto mb-2 text-blue-500" />
            <div className="text-2xl font-bold text-gray-900">{result.document_count}</div>
            <div className="text-sm text-gray-600">Documents</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <Layers className="h-6 w-6 mx-auto mb-2 text-green-500" />
            <div className="text-2xl font-bold text-gray-900">1</div>
            <div className="text-sm text-gray-600">Unified Summary</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <Clock className="h-6 w-6 mx-auto mb-2 text-purple-500" />
            <div className="text-2xl font-bold text-gray-900">~2</div>
            <div className="text-sm text-gray-600">Minutes Saved</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <AlertTriangle className="h-6 w-6 mx-auto mb-2 text-orange-500" />
            <div className="text-2xl font-bold text-gray-900">Auto</div>
            <div className="text-sm text-gray-600">Conflict Detection</div>
          </CardContent>
        </Card>
      </div>

      {/* Unified Summary - Main Feature */}
      <Card className="border-green-200">
        <CardHeader className="bg-green-50">
          <CardTitle className="flex items-center text-green-800">
            <Layers className="h-5 w-5 mr-2" />
            Unified Summary
          </CardTitle>
          <p className="text-sm text-green-700">
            AI-generated synthesis of all documents with theme organization
          </p>
        </CardHeader>
        <CardContent className="p-6">
          <div className="prose prose-gray max-w-none">
            <p className="text-gray-700 leading-relaxed whitespace-pre-line">
              {result.unified_summary}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Individual Summaries */}
      <div>
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <FileText className="h-5 w-5 mr-2 text-blue-500" />
          Individual Document Summaries
        </h3>
        <div className="space-y-4">
          {result.individual_summaries.map((item, index) => (
            <Card key={index} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center justify-between">
                  <div className="flex items-center">
                    <FileText className="h-4 w-4 mr-2 text-blue-500" />
                    {item.filename}
                  </div>
                  <span className="text-xs text-gray-500 font-normal">
                    {(item.original_size / 1024).toFixed(1)} KB
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <p className="text-gray-700 leading-relaxed">{item.summary}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Next Steps Teaser */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="p-6">
          <h4 className="font-semibold text-blue-900 mb-2">🚀 Coming Tomorrow (Day 2):</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• PDF & DOCX file support</li>
            <li>• Source attribution (click to see original text)</li>
            <li>• Theme-based organization</li>
            <li>• Enhanced contradiction detection</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}