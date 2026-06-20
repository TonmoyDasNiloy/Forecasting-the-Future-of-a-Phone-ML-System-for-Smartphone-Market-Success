"use client"

import { useTheme } from "@/components/theme-provider"

function DarkLogo({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 300 300" fill="none"
      xmlns="http://www.w3.org/2000/svg" role="img" aria-label="BI Agent logo">
      <defs>
        <linearGradient id="d-bar1" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0" stopColor="#6366F1"/><stop offset="1" stopColor="#818CF8"/>
        </linearGradient>
        <linearGradient id="d-bar2" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0" stopColor="#3B82F6"/><stop offset="1" stopColor="#38BDF8"/>
        </linearGradient>
        <linearGradient id="d-bar3" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0" stopColor="#22C55E"/><stop offset="1" stopColor="#4ADE80"/>
        </linearGradient>
        {/* Glow filter for dark mode */}
        <filter id="d-glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="4" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <clipPath id="d-ringClip"><circle cx="150" cy="150" r="96"/></clipPath>
      </defs>

      {/* Background */}
      <rect width="300" height="300" rx="64" fill="#0A0C14"/>

      {/* Phone body — thicker stroke, more visible */}
      <rect x="84" y="28" width="132" height="244" rx="30"
        fill="#161B2E" stroke="#3D4F8A" strokeWidth="5"/>
      {/* Notch — bigger */}
      <rect x="128" y="40" width="44" height="8" rx="4" fill="#3D4F8A"/>
      {/* Home bar */}
      <rect x="130" y="266" width="40" height="6" rx="3" fill="#3D4F8A"/>

      {/* Radar ring — thicker, brighter */}
      <circle cx="150" cy="150" r="96"
        fill="none" stroke="#6366F1" strokeWidth="10" opacity="0.95"
        filter="url(#d-glow)"/>

      {/* Sweep arcs — thicker */}
      <path d="M106 72 A90 90 0 0 1 180 65"
        fill="none" stroke="#818CF8" strokeWidth="11" strokeLinecap="round" opacity="0.9"/>
      <path d="M92 94 A78 78 0 0 1 158 72"
        fill="none" stroke="#818CF8" strokeWidth="10" strokeLinecap="round" opacity="0.65"/>

      {/* Bars — wider, taller for visibility */}
      <g clipPath="url(#d-ringClip)">
        <rect x="95" y="152" width="34" height="58" rx="9" fill="url(#d-bar1)"/>
        <rect x="133" y="118" width="34" height="92" rx="9" fill="url(#d-bar2)"/>
        <rect x="171" y="88" width="34" height="122" rx="9" fill="url(#d-bar3)"/>
      </g>

      {/* Trend line — thicker */}
      <line x1="104" y1="156" x2="212" y2="78"
        stroke="#F59E0B" strokeWidth="8" strokeLinecap="round"/>
      {/* Amber dot — bigger */}
      <circle cx="212" cy="78" r="13" fill="#F59E0B"/>
      <circle cx="212" cy="78" r="6" fill="white" opacity="0.5"/>
    </svg>
  )
}

function LightLogo({ size }: { size: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 300 300" fill="none"
      xmlns="http://www.w3.org/2000/svg" role="img" aria-label="BI Agent logo">
      <defs>
        <linearGradient id="l-bar1" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0" stopColor="#6366F1"/><stop offset="1" stopColor="#818CF8"/>
        </linearGradient>
        <linearGradient id="l-bar2" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0" stopColor="#3B82F6"/><stop offset="1" stopColor="#38BDF8"/>
        </linearGradient>
        <linearGradient id="l-bar3" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0" stopColor="#22C55E"/><stop offset="1" stopColor="#4ADE80"/>
        </linearGradient>
        <clipPath id="l-ringClip"><circle cx="150" cy="150" r="96"/></clipPath>
      </defs>

      {/* Background */}
      <rect width="300" height="300" rx="64" fill="#F8FAFF" stroke="#C7D2FE" strokeWidth="3"/>

      {/* Phone body — thicker stroke */}
      <rect x="84" y="28" width="132" height="244" rx="30"
        fill="#EEF2FF" stroke="#A5B4FC" strokeWidth="5"/>
      {/* Notch */}
      <rect x="128" y="40" width="44" height="8" rx="4" fill="#A5B4FC"/>
      {/* Home bar */}
      <rect x="130" y="266" width="40" height="6" rx="3" fill="#A5B4FC"/>

      {/* Radar ring — thicker */}
      <circle cx="150" cy="150" r="96"
        fill="none" stroke="#6366F1" strokeWidth="10"/>

      {/* Sweep arcs — thicker */}
      <path d="M106 72 A90 90 0 0 1 180 65"
        fill="none" stroke="#6366F1" strokeWidth="11" strokeLinecap="round" opacity="0.75"/>
      <path d="M92 94 A78 78 0 0 1 158 72"
        fill="none" stroke="#6366F1" strokeWidth="10" strokeLinecap="round" opacity="0.5"/>

      {/* Bars — wider */}
      <g clipPath="url(#l-ringClip)">
        <rect x="95" y="152" width="34" height="58" rx="9" fill="url(#l-bar1)"/>
        <rect x="133" y="118" width="34" height="92" rx="9" fill="url(#l-bar2)"/>
        <rect x="171" y="88" width="34" height="122" rx="9" fill="url(#l-bar3)"/>
      </g>

      {/* Trend line — thicker */}
      <line x1="104" y1="156" x2="212" y2="78"
        stroke="#F59E0B" strokeWidth="8" strokeLinecap="round"/>
      {/* Amber dot — bigger */}
      <circle cx="212" cy="78" r="13" fill="#F59E0B"/>
      <circle cx="212" cy="78" r="6" fill="white" opacity="0.6"/>
    </svg>
  )
}

export function BIAgentLogo({ size = 40 }: { size?: number }) {
  const { theme } = useTheme()
  return theme === "dark" ? <DarkLogo size={size} /> : <LightLogo size={size} />
}
