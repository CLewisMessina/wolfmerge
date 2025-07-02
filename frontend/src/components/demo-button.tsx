// frontend/src/components/demo-button.tsx

"use client"

import { Button } from '@/components/ui/button'
import { Sparkles } from 'lucide-react'

interface DemoButtonProps {
  onLoadDemo: () => void
  isDisabled: boolean
}

export function DemoButton({ onLoadDemo, isDisabled }: DemoButtonProps) {
  return (
    <div className="text-center py-6">
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-6 border border-purple-100">
        <p className="text-gray-700 mb-4">
          Don't have documents ready? See WolfMerge in action:
        </p>
        <Button 
          variant="outline" 
          onClick={onLoadDemo}
          disabled={isDisabled}
          className="bg-gradient-to-r from-purple-500 to-blue-600 text-white border-0 hover:opacity-90 transition-opacity"
        >
          <Sparkles className="h-4 w-4 mr-2" />
          Try Sample Documents
        </Button>
        <p className="text-xs text-gray-500 mt-3">
          Includes: Meeting notes, project proposal, and budget overview with realistic conflicts
        </p>
      </div>
    </div>
  )
}