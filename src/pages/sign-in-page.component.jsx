import Logo from '../assets/logo.png'
import SignIn from '../components/sign-in.component'

export default function SignUpPage() {
  return (
    <div className="relative bg-gray-800 overflow-hidden">
      <div className="hidden sm:block sm:absolute sm:inset-0" aria-hidden="true">
        <svg
          className="absolute bottom-0 right-0 transform translate-x-1/2 mb-48 text-gray-700 lg:top-0 lg:mt-28 lg:mb-0 xl:transform-none xl:translate-x-0"
          width={364}
          height={384}
          viewBox="0 0 364 384"
          fill="none"
        >
          <defs>
            <pattern
              id="eab71dd9-9d7a-47bd-8044-256344ee00d0"
              x={0}
              y={0}
              width={20}
              height={20}
              patternUnits="userSpaceOnUse"
            >
              <rect x={0} y={0} width={4} height={4} fill="currentColor" />
            </pattern>
          </defs>
          <rect width={364} height={384} fill="url(#eab71dd9-9d7a-47bd-8044-256344ee00d0)" />
        </svg>
      </div>
      <div className="relative pt-6 pb-16 sm:pb-24">
          <nav
            className="relative max-w-7xl mx-auto flex items-center justify-between px-4 sm:px-6"
            aria-label="Global"
          >
            <div className="flex items-center flex-1">
              <div className="flex items-center justify-between w-full md:w-auto">
                <a href="/" className="flex">
                  <img
                    className="w-auto h-10 mr-2"
                    src={Logo}
                    alt="Manentia Advisory"
                  />
                  <div className="m-auto text-white text-lg font-base tracking-wide subpixel-antialiased">Manetia Advisory</div>
                </a>
              </div>
            </div>
            <div className="flex">
              <a href="/"
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-gray-600 hover:bg-gray-700">
                Contact Us
              </a>
            </div>
          </nav>

        <main className="mt-16 sm:mt-24">
          <div className="mx-auto max-w-7xl">
            <div className="lg:grid lg:grid-cols-12 lg:gap-8">
              <div className="px-10 sm:text-center md:max-w-2xl md:mx-auto lg:col-span-6 lg:text-left lg:flex lg:items-center">
                <div>
                  <h1 className="mt-4 lg:mb-9 text-4xl tracking-tight font-extrabold text-white sm:mt-5 sm:leading-none lg:mt-6 lg:text-5xl xl:text-6xl">
                    <span className="md:block">AI Diagnosis <span className="lg:block lg:text-red-400 text-white">Tele-Screening</span></span>
                    <p className="lg:text-white text-red-400 md:block">Beta Testing</p>
                  </h1>
                  <p className="mt-3 mb-4 sm:mb-10 text-base text-gray-300 sm:mt-5 sm:text-xl lg:text-xl xl:text-2xl ">
                    Equipped with Machine Learning and the latest technologies to provide rapid diagonsis reports.
                  </p>
                  <a href="/" className="text-red-400 text-base sm:text-xl underline">Want to join beta testing?</a>
                </div>
              </div>
              <div className="mt-16 sm:mt-24 lg:mt-0 mx-10 sm:mx-0 lg:col-span-6">
                <SignIn />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
