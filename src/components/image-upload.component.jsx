import React from 'react'

class ImageUplaod extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
        image: ''
        }
    }

    render() {
        return (
            <div className="mx-6 sm:mx-16 mb-8 bg-white shadow overflow-hidden rounded-lg border border-gray-300">
                <div className="px-4 py-5 sm:px-6">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">Test X-Ray Image</h3>
                    <p className="mt-1 text-sm text-gray-500">Currently Patient information fields are disabled. Please <span className="font-bold">upload only original x-ray image</span>, as the system is development phase, might give unappropritate result.</p>
                </div>
                <div className="border-t border-gray-200 px-4 py-5 sm:px-6">
                    <dl className="grid gap-x-4 gap-y-8 grid-cols-2">
                        <div className="col-span-1">
                            <dt className="text-sm font-medium text-gray-500">Full name</dt>
                            <dd className="mt-1 text-sm text-gray-500 py-1.5 px-2 border border-gray-300 rounded-md select-none">Margot Foster</dd>
                        </div>
                        <div className="col-span-1">
                            <dt className="text-sm font-medium text-gray-500">Age</dt>
                            <dd className="mt-1 text-sm text-gray-500 py-1.5 px-2 border border-gray-300 rounded-md select-none">36 Years</dd>
                        </div>
                        <div className="col-span-1">
                            <dt className="text-sm font-medium text-gray-500">Weight</dt>
                            <dd className="mt-1 text-sm text-gray-500 py-1.5 px-2 border border-gray-300 rounded-md select-none">84 Kg</dd>
                        </div>
                        <div className="col-span-1">
                            <dt className="text-sm font-medium text-gray-500">Gender</dt>
                            <dd className="mt-1 text-sm text-gray-500 py-1.5 px-2 border border-gray-300 rounded-md select-none">Male</dd>
                        </div>
                        <div className="col-span-2">
                            <div className="mx-1">
                                <label className="block text-sm font-medium text-gray-600">Upload x-ray image</label>
                                <div className="mt-1 flex justify-center px-6 pt-8 pb-10 border-2 border-gray-300 border-dashed rounded-md">
                                <div className="space-y-1 text-center">
                                    <svg
                                    className="mx-auto h-12 w-12 text-gray-400"
                                    stroke="currentColor"
                                    fill="none"
                                    viewBox="0 0 48 48"
                                    aria-hidden="true"
                                    >
                                    <path
                                        d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                                        strokeWidth={2}
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                    </svg>
                                    <div className="flex text-sm text-gray-600">
                                        <label htmlFor="file-upload"
                                            className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                                            <span>Upload a file</span>
                                            <input id="file-upload" name="file-upload" type="file" className="sr-only" />
                                        </label>
                                        <p className="pl-1">or drag and drop</p>
                                    </div>
                                    <p className="text-xs text-gray-500">PNG and JPG up to 20MB</p>
                                </div>
                                </div>
                            </div>
                        </div>
                    </dl>
                </div>
            </div>

        )
    }
}

export default ImageUplaod