// src\components\file-upload.tsx
"use client"

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Upload, FileText, X, AlertCircle } from 'lucide-react'

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void
  onAnalyze: () => void
  selectedFiles: File[]
  isAnalyzing: boolean
}

export function FileUpload({ onFilesSelected, onAnalyze, selectedFiles, isAnalyzing }: FileUploadProps) {
  const [error, setError] = useState<string>("")

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError("")
    
    // Handle rejected files
    if (rejectedFiles.length > 0) {
      setError("Only .txt and .md files are supported")
      return
    }

    // Limit to 3 files total
    const totalFiles = selectedFiles.length + acceptedFiles.length
    if (totalFiles > 3) {
      setError("Maximum 3 files allowed")
      return
    }

    // Add new files to existing selection
    const newFiles = [...selectedFiles, ...acceptedFiles].slice(0, 3)
    onFilesSelected(newFiles)
  }, [selectedFiles, onFilesSelected])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt'],
      'text/markdown': ['.md']
    },
    maxFiles: 3,
    disabled: isAnalyzing
  })

  const removeFile = (index: number) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index)
    onFilesSelected(newFiles)
    setError("")
  }

  return (
    <div className="space-y-4">
      <Card 
        {...getRootProps()} 
        className={`border-2 border-dashed p-8 text-center cursor-pointer transition-colors ${
          isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : isAnalyzing 
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className={`mx-auto h-12 w-12 mb-4 ${isAnalyzing ? 'text-gray-300' : 'text-gray-400'}`} />
        <h3 className="text-lg font-semibold mb-2">
          {isDragActive ? 'Drop files here...' : 'Upload Your Documents'}
        </h3>
        <p className="text-gray-600 mb-4">
          Drag & drop up to 3 files, or click to browse
        </p>
        <p className="text-sm text-gray-500">
          Supports: .txt, .md files only
        </p>
        <p className="text-xs text-blue-600 mt-2">
          📅 Coming tomorrow: PDF & DOCX support!
        </p>
      </Card>

      {error && (
        <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {selectedFiles.length > 0 && (
        <div className="space-y-2">
          <h4 className="font-medium">Selected Files ({selectedFiles.length}/3):</h4>
          {selectedFiles.map((file, index) => (
            <div key={index} className="flex items-center justify-between bg-gray-50 p-3 rounded-lg">
              <div className="flex items-center">
                <FileText className="h-4 w-4 mr-2 text-blue-500" />
                <span className="text-sm font-medium">{file.name}</span>
                <span className="text-xs text-gray-500 ml-2">
                  ({(file.size / 1024).toFixed(1)} KB)
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removeFile(index)}
                disabled={isAnalyzing}
                className="hover:bg-red-50 hover:text-red-600"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          ))}
          
          <Button 
            onClick={onAnalyze} 
            disabled={selectedFiles.length === 0 || isAnalyzing}
            className="w-full mt-4 bg-blue-600 hover:bg-blue-700"
          >
            {isAnalyzing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Analyzing Documents...
              </>
            ) : (
              'Analyze Documents'
            )}
          </Button>
        </div>
      )}
    </div>
  )
}