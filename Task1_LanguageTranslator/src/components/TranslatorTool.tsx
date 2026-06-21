import { useState, useCallback, useEffect } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Languages, Copy, Check, Volume2, ArrowRightLeft, Loader2, AlertCircle, Sparkles } from 'lucide-react'
const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ru', name: 'Russian' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi', name: 'Hindi' },
  { code: 'nl', name: 'Dutch' },
  { code: 'sv', name: 'Swedish' },
  { code: 'pl', name: 'Polish' },
  { code: 'tr', name: 'Turkish' },
  { code: 'vi', name: 'Vietnamese' },
  { code: 'th', name: 'Thai' },
  { code: 'id', name: 'Indonesian' },
  { code: 'uk', name: 'Ukrainian' },
]
export default function TranslatorTool() {
  const [inputText, setInputText] = useState('')
  const [translatedText, setTranslatedText] = useState('')
  const [sourceLang, setSourceLang] = useState('en')
  const [targetLang, setTargetLang] = useState('es')
  const [isTranslating, setIsTranslating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const charCount = inputText.length
  const maxChars = 5000
  const translateText = useCallback(async () => {
    if (!inputText.trim()) { setError('Please enter text to translate'); return }
    setIsTranslating(true); setError(null); setTranslatedText('')
    try {
      const langPair = ${sourceLang}|${targetLang}
      const url = https://api.mymemory.translated.net/get?q=${encodeURIComponent(inputText)}&langpair=${langPair}
      const response = await fetch(url)
      const data = await response.json()
      if (data.responseStatus === 200 && data.responseData?.translatedText) {
        setTranslatedText(data.responseData.translatedText)
      } else { throw new Error(data.responseDetails || 'Translation failed') }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to translate. Please try again.')
    } finally { setIsTranslating(false) }
  }, [inputText, sourceLang, targetLang])
  const swapLanguages = useCallback(() => {
    setSourceLang(targetLang); setTargetLang(sourceLang)
    if (translatedText) { setInputText(translatedText); setTranslatedText(inputText) }
  }, [sourceLang, targetLang, inputText, translatedText])
  const copyToClipboard = useCallback(async () => {
    if (translatedText) { await navigator.clipboard.writeText(translatedText); setCopied(true); setTimeout(() => setCopied(false), 2000) }
  }, [translatedText])
  const speakText = useCallback((text: string, lang: string) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = lang; utterance.rate = 0.9
      utterance.onstart = () => setIsSpeaking(true)
      utterance.onend = () => setIsSpeaking(false)
      utterance.onerror = () => setIsSpeaking(false)
      window.speechSynthesis.speak(utterance)
    }
  }, [])
  useEffect(() => { return () => { if ('speechSynthesis' in window) window.speechSynthesis.cancel() } }, [])
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl shadow-lg">
              <Languages className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Language Translator
          </h1>
          <p className="text-gray-600">Translate text between 20+ languages instantly</p>
        </div>
        <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm">
          <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Source Language</label>
                <Select value={sourceLang} onValueChange={setSourceLang}>
                  <SelectTrigger className="h-12"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {LANGUAGES.map((lang) => (<SelectItem key={lang.code} value={lang.code}>{lang.name}</SelectItem>))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-end justify-center md:justify-end pb-2">
                <Button variant="outline" size="icon" onClick={swapLanguages} className="rounded-full h-10 w-10 border-2 hover:bg-blue-50 hover:border-blue-300 transition-all">
                  <ArrowRightLeft className="w-4 h-4" />
                </Button>
              </div>
              <div className="space-y-2 md:col-start-2">
                <label className="text-sm font-medium text-gray-700">Target Language</label>
                <Select value={targetLang} onValueChange={setTargetLang}>
                  <SelectTrigger className="h-12"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {LANGUAGES.map((lang) => (<SelectItem key={lang.code} value={lang.code}>{lang.name}</SelectItem>))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <Separator className="my-6" />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-gray-700">Input Text</label>
                  <Badge variant="secondary" className={charCount > maxChars ? 'bg-red-100 text-red-700' : ''}>
                    {charCount.toLocaleString()} / {maxChars.toLocaleString()}
                  </Badge>
                </div>
                <div className="relative">
                  <Textarea value={inputText} onChange={(e) => setInputText(e.target.value.slice(0, maxChars))} placeholder="Enter text to translate..." className="min-h-[200px] resize-none text-base leading-relaxed" maxLength={maxChars} />
                  {inputText && (
                    <Button variant="ghost" size="sm" onClick={() => speakText(inputText, sourceLang)} disabled={isSpeaking} className="absolute bottom-2 right-2 h-8 w-8 p-0">
                      <Volume2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-gray-700">Translation</label>
                  {translatedText && (
                    <Button variant="ghost" size="sm" onClick={copyToClipboard} className="h-8 gap-1">
                      {copied ? (<><Check className="w-4 h-4 text-green-500" /><span className="text-green-600">Copied!</span></>) : (<><Copy className="w-4 h-4" /><span>Copy</span></>)}
                    </Button>
                  )}
                </div>
                <div className="relative">
                  <Textarea value={isTranslating ? 'Translating...' : translatedText} readOnly placeholder="Translation will appear here..." className="min-h-[200px] resize-none text-base leading-relaxed bg-gray-50" />
                  {translatedText && (
                    <Button variant="ghost" size="sm" onClick={() => speakText(translatedText, targetLang)} disabled={isSpeaking} className="absolute bottom-2 right-2 h-8 w-8 p-0">
                      <Volume2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </div>
            </div>
            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                <div><p className="font-medium text-red-800">Translation Error</p><p className="text-sm text-red-600">{error}</p></div>
              </div>
            )}
            <div className="mt-6 flex justify-center">
              <Button onClick={translateText} disabled={isTranslating || !inputText.trim()} size="lg" className="px-8 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white shadow-lg hover:shadow-xl transition-all">
                {isTranslating ? (<><Loader2 className="w-5 h-5 mr-2 animate-spin" />Translating...</>) : (<><Sparkles className="w-5 h-5 mr-2" />Translate</>)}
              </Button>
            </div>
          </CardContent>
        </Card>
        <div className="mt-6 text-center text-sm text-gray-500">
          <p>Powered by MyMemory Translation API - Free translation service</p>
        </div>
      </div>
    </div>
  )
}