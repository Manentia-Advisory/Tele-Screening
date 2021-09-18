import React from 'react'
import { Link } from 'react-router-dom'

class SignIn extends React.Component {

    constructor(props) {
        super(props);

        this.state = {
            username: '',
            password: '',
        }
    }

    handleSubmit = () => {

    }

    handleChange = () => {
      
    }

    render() {
        return (
            <div className="bg-white sm:max-w-md sm:w-full sm:mx-auto rounded-lg sm:overflow-hidden">
                  <div className="px-4 py-8 sm:px-10">
                    <div>
                      <p className="text-lg sm:text-2xl font-medium text-gray-700">Sign in to testing account</p>
                    </div>

                    <div className="mt-6">
                      <form className="space-y-6" onSubmit={this.handleSubmit}>

                        <div>
                          <label htmlFor="Username" className="block text-sm font-medium text-gray-700">
                            Username
                          </label>
                          <input
                            type="text"
                            name="username"
                            id="username"
                            autoComplete="username"
                            placeholder="Enter provided username"
                            value={this.state.email}
                            onChange={this.handleChange}
                            required
                            className="block mt-1 w-full shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm border-gray-300 rounded-md"
                          />
                        </div>

                        <div>
                          <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                            Password
                          </label>
                          <input
                            id="password"
                            name="password"
                            type="password"
                            placeholder="Password"
                            autoComplete="current-password"
                            value={this.state.password}
                            onChange={this.handleChange}
                            required
                            className="block mt-1 w-full shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm border-gray-300 rounded-md"
                          />
                        </div>

                        <div className="flex items-center justify-between">
                        
                          <div className="flex items-center">
                              <input
                              id="remember-me"
                              name="remember-me"
                              type="checkbox"
                              className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                              />
                              <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-900">
                              Remember me
                              </label>
                          </div>

                          <div className="text-sm text-right">
                              <a href="/" className="font-medium text-indigo-600 hover:text-indigo-500">
                              Forgot username or password?
                              </a>
                          </div>
                        </div>

                        <div>
                          <Link to='/app'>
                            <button type="submit"
                                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                                Sign in
                            </button>
                          </Link>
                        </div>
                      </form>
                    </div>
                  </div>
                  <div className="px-4 py-6 bg-gray-50 border-t-2 rounded-b-lg border-gray-200 sm:px-10">
                    <p className="text-xs leading-5 text-gray-500">
                      By signing up, you agree to our{' '}
                      <a href="/" className="font-medium text-gray-900 hover:underline">
                        Terms
                      </a>
                      ,{' '}
                      <a href="/" className="font-medium text-gray-900 hover:underline">
                        Data Policy
                      </a>{' '}
                      and{' '}
                      <a href="/" className="font-medium text-gray-900 hover:underline">
                        Cookies Policy
                      </a>
                      .
                    </p>
                  </div>
                </div>
        )
    }
}

export default SignIn;