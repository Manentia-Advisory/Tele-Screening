import { Link } from 'react-router-dom'

import { Disclosure } from '@headlessui/react'
import { MenuIcon, XIcon} from '@heroicons/react/outline'

import Logo from '../assets/logo.png'

const userNavigation = [
  { name: 'Contact Us', href: '/' },
  { name: 'Sign Out', href: '/' },
]

export default function NavBar() {
  return (
    <Disclosure as="nav" className="bg-blue-900">
      {({ open }) => (
        <>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex">
                  <div className="flex-shrink-0 flex items-center">
                  <img
                    className="block mt-1 h-10 w-auto"
                    src={Logo}
                    alt="Manentia Advisory"
                  />
                  <span className="text-white text-lg ml-2 tracking-wide subpixel-antialiased">Manentia Advisory</span>
                </div>
              </div>
              <div className="flex items-center">
                <div className="flex-shrink-0 hidden md:block">
                  <button
                    type="button"
                    className="relative inline-flex items-center px-4 py-2 border border-white shadow-sm text-sm font-semibold rounded-md text-white hover:text-blue-900 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500"
                  >
                    <span>Contact Us</span>
                    
                  </button>
                </div>
                
                <div className="hidden md:ml-4 md:flex-shrink-0 md:flex md:items-center">
                <div className="flex-shrink-0 hidden md:block">
                  <Link to='/'>
                    <button type="button"
                      className="relative inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-semibold rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500">
                      <span>Sign Out</span>
                    </button>
                  </Link>
                </div>
                </div>

                <div className="ml-2 -mr-2 flex items-center md:hidden">
                  {/* Mobile menu button */}
                  <Disclosure.Button className="inline-flex items-center justify-center p-2 rounded-md text-white hover:text-blue-900 hover:bg-white focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white">
                    <span className="sr-only">Open main menu</span>
                    {open ? (
                      <XIcon className="block h-6 w-6" aria-hidden="true" />
                    ) : (
                      <MenuIcon className="block h-6 w-6" aria-hidden="true" />
                    )}
                  </Disclosure.Button>
                </div>
                
              </div>
            </div>
          </div>

          <Disclosure.Panel className="md:hidden">
            <div className="pb-3 border-t border-white">
              <div className="mt-3 px-2 space-y-1 sm:px-3">
                {userNavigation.map((item) => (
                  <a
                    key={item.name}
                    href={item.href}
                    className="block px-3 py-2 rounded-md text-base font-medium text-white hover:text-blue-900 hover:bg-white"
                  >
                    {item.name}
                  </a>
                ))}
              </div>
            </div>
          </Disclosure.Panel>
        </>
      )}
    </Disclosure>
  )
}
